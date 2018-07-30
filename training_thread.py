# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from utils.accum_trainer import AccumTrainer
from scene_loader import THORDiscreteEnvironment as Environment
from network import ActorCriticFFNetwork

from constants import ACTION_SIZE
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import VERBOSE
from constants import USE_LSTM
import pdb

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device,
               network_scope="network",
               scene_scope="scene",
               task_scope="task"):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    self.network_scope = network_scope
    self.scene_scope = scene_scope
    self.task_scope = task_scope
    self.scopes = [network_scope, scene_scope, task_scope]

    self.local_network = ActorCriticFFNetwork(
                           action_size=ACTION_SIZE,
                           device=device,
                           network_scope=network_scope,
                           scene_scopes=[scene_scope])

    

    self.local_network.prepare_loss(ENTROPY_BETA, self.scopes)


    self.trainer = AccumTrainer(device)
    self.trainer.prepare_minimize(self.local_network.total_loss,
                                  self.local_network.get_vars())

    new_variable_list=self.local_network.get_vars()
    old_varaible_list=self.local_network.get_vars_old()

    '''
    here we can have an assign op
    '''


    


    self.accum_gradients = self.trainer.accumulate_gradients() #This is to assign gradients into the 

    
    self.reset_gradients = self.trainer.reset_gradients() #after applying the grads to variables we need to resent those variables


    accum_grad_names = [self._local_var_name(x) for x in self.trainer.get_accum_grad_list()] #get the name list of all the grad vars

    global_net_vars = [x for x in global_network.get_vars() if self._get_accum_grad_name(x) in accum_grad_names] #check whether the global_network vars are mentioned in gradiet computations for them

    #self.trainer.get_accum_grad_list() this is about gradients righjt now

    self.apply_gradients = grad_applier.apply_gradients(        #appying the gradiets  like the op (we plly gradient to the master network)
      global_net_vars, self.trainer.get_accum_grad_list() )

    #This is very important here from the local network gradients we directly update the GLOBAL network :) That is called Asyncronous    

    self.sync = self.local_network.sync_from(global_network) #this is to sync from the glocal network Apply updated global params to the local network
    


    self.env = None

    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0
    self.episode_length = 0
    self.episode_max_q = -np.inf

  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def _get_accum_grad_name(self, var):
    return self._local_var_name(var).replace(':','_') + '_accum_grad:0'

  def _anneal_learning_rate(self, global_time_step):
    time_step_to_go = max(self.max_global_time_step - global_time_step, 0.0)
    learning_rate = self.initial_learning_rate * time_step_to_go / self.max_global_time_step
    return learning_rate

  def choose_action(self, pi_values):
    values = []
    sum = 0.0
    for rate in pi_values:
      sum = sum + rate
      value = sum
      values.append(value)

    r = random.random() * sum
    for i in range(len(values)):
      if values[i] >= r:
        return i

    # fail safe
    return len(values) - 1

  def _record_score(self, sess, writer, summary_op, placeholders, values, global_t):
    feed_dict = {}
    for k in placeholders:
      feed_dict[placeholders[k]] = values[k]
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    if VERBOSE: print('writing to summary writer at time %d\n' % (global_t))
    writer.add_summary(summary_str, global_t)
    # writer.flush()

  def process(self, sess, global_t, summary_writer, summary_op, summary_placeholders):

    if self.env is None:
      # lazy evaluation
      time.sleep(self.thread_index*1.0)
      self.env = Environment({
        'scene_name': self.scene_scope,
        'terminal_state_id': int(self.task_scope)
      })
      self.env.reset() #resetting the environment for each thread

    states = [] #to keeep state ,actions ,targets and other stae
    actions = []
    rewards = []
    values = []
    targets = []

    terminal_end = False

    # reset accumulated gradients
    sess.run( self.reset_gradients ) #resetting the gradient positions when starting the process for each 

    # copy weights from shared to local
    sess.run(self.sync)

    start_local_t = self.local_t

    # t_max times loop
    for i in range(LOCAL_T_MAX):
      
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.env.s_t, self.env.target, self.scopes)
      #pi_Old, value_Old = self.local_network.run_policy_and_value_old(sess, self.env.s_t, self.env.target, self.scopes)
      
      
      action = self.choose_action(pi_)

      

      states.append(self.env.s_t) 
      actions.append(action)
      values.append(value_)
      targets.append(self.env.target)

      if VERBOSE and (self.thread_index == 0) and (self.local_t % 1000) == 0:
        sys.stdout.write("Pi = {0} V = {1}\n".format(pi_, value_))

      # process game
      self.env.step(action)

      

      # receive game result
      reward = self.env.reward  #getting the reward from the env
      terminal = self.env.terminal #geting whether the agent went to a terminal state

      # ad-hoc reward for navigation
      reward = 10.0 if terminal else -0.01 #this is the normal reward here 10 if terminal all the others it is -0.01 (ollision donesst take in to the accout)
      if self.episode_length > 5e3: terminal = True #Here we do not let agent to run more that 5000 steps so we make it terminal
      #but the above terminal thing has no effect on giving 10 as the rwaerd because we set the rweard above

      self.episode_reward += reward
      self.episode_length += 1
      #this is what is the maximum value got in the episode
      self.episode_max_q = max(self.episode_max_q, np.max(value_)) #self.episode_max_q-This is -inf in the beggining 

      # clip reward
      rewards.append(np.clip(reward, -1, 1)) #make sure the rewartds is between -1 and +1 even thore rtthere is a 10
      

      self.local_t += 1

      # s_t1 -> s_t
      self.env.update()

      if terminal:
        print("Terminated Good")
        terminal_end = True
        sys.stdout.write("time %d | thread #%d | scene %s | target #%s\n%s %s episode reward = %.3f\n%s %s episode length = %d\n%s %s episode max Q  = %.3f\n" % (global_t, self.thread_index, self.scene_scope, self.task_scope, self.scene_scope, self.task_scope, self.episode_reward, self.scene_scope, self.task_scope, self.episode_length, self.scene_scope, self.task_scope, self.episode_max_q))

        summary_values = {
          "episode_reward_input": self.episode_reward,
          "episode_length_input": float(self.episode_length),
          "episode_max_q_input": self.episode_max_q,
          "learning_rate_input": self._anneal_learning_rate(global_t)
        }

        self._record_score(sess, summary_writer, summary_op, summary_placeholders,
                           summary_values, global_t)
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q = -np.inf #after a terminaltion we do this
        self.env.reset()

        break

    R = 0.0 #In the terminal Return is nothing 
    if not terminal_end: #But if it's not the turminal Return is the next value function
      print("Yahooo Return is not next value function we break the loop before a terminal state")
      R = self.local_network.run_value(sess, self.env.s_t, self.env.target, self.scopes)

    Returns=np.zeros_like(rewards)
    Advants=np.zeros_like(rewards)
    lastgaelam=0
    LAMBDA=0.99

    self.nsteps=len(rewards)

    ############################################################################# we should assined all params to the new params

    #This will only has an effect on 

    #####################################################################

    for t in reversed(range(self.nsteps)):
      if t==self.nsteps-1:
        nextvalues = R

      else:
        nextvalues = values[t+1] 
      delta = rewards[t] + GAMMA * nextvalues  - values[t]
      Advants[t] = lastgaelam = delta + GAMMA * LAMBDA * lastgaelam

    Returns=Advants+values #This is more of the v_next

    Advants=(Advants - Advants.mean()) / (Advants.std() + 1e-8)
      


    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []
    batch_t = []

    

    # compute and accmulate gradients
    for(ai, ri, si, Vi, ti) in zip(actions, rewards, states, values, targets):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1 #making the actions one hot

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)
      batch_t.append(ti)



   
    sess.run( self.accum_gradients, #since we update the algorithm for given action ,given state, given advatns and given value and given reward we do not care about the sequence
              feed_dict = {
                self.local_network.s: batch_si,
                self.local_network.a: batch_a,
                self.local_network.t: batch_t,
                self.local_network.td: batch_td,
                self.local_network.r: batch_R} )

    cur_learning_rate = self._anneal_learning_rate(global_t)

    sess.run( self.apply_gradients,
              feed_dict = { self.learning_rate_input: cur_learning_rate } )

    if VERBOSE and (self.thread_index == 0) and (self.local_t % 100) == 0:
      sys.stdout.write("Local timestep %d\n" % self.local_t)

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t

