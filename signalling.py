import numpy as np

class Sender:
        
    def __init__(self, n_inputs: int, n_messages: int, eps: float = 1e-6):
        self.n_messages = n_messages
        self.message_weights = np.zeros((n_inputs, n_messages))
        self.message_weights.fill(eps)
        self.last_situation = (0, 0)
        
    def send_message(self, input: int) -> int:
        probs = np.exp(self.message_weights[input, :])/np.sum(np.exp(self.message_weights[input, :]))
        message = np.random.choice(self.n_messages, p=probs)
        self.last_situation = (input, message)
        return message

    def learn_from_feedback(self, reward: int) -> None:
        self.message_weights[self.last_situation] += reward


class Receiver:
        
    def __init__(self, n_messages: int, n_actions: int, eps: float = 1e-6):
        self.n_actions = n_actions
        self.action_weights = np.ndarray((n_messages, n_actions))
        self.action_weights.fill(eps)
        self.last_situation = (0, 0)
        
    def act(self, message: int) -> int:
        probs = np.exp(self.action_weights[message, :])/np.sum(np.exp(self.action_weights[message, :]))
        action = np.random.choice(self.n_actions, p=probs)
        self.last_situation = (message, action)
        return action

    def learn_from_feedback(self, reward: int) -> None:
        self.action_weights[self.last_situation] += reward


class World:
    def __init__(self, n_states: int, seed: int = 42):
        self.n_states = n_states
        self.state = 0
        self.rng = np.random.RandomState(seed)
        
    def emit_state(self) -> int:
        self.state = self.rng.randint(self.n_states)
        return self.state
    
    def evaluate_action(self, action: int) -> int:
        return 1 if action == self.state else -1


class Game :
    def __init__(self, n_states: int, re_config, se_config ,seed: int = 42, eps: float = 1e-6):
     self.world = World(n_states, seed=seed)
     self.receiver = Receiver(n_messages=re_config[0], n_actions=re_config[1], eps=eps)
     self.sender =  Sender(n_inputs=se_config[0], n_messages=se_config[1], eps=eps)

    def train(self, eval_size, threshold):
        eval_rewards =  0
        epoch = 0
        while True :

            world_state = self.world.emit_state()
            message = self.sender.send_message(world_state)
            action = self.receiver.act(message)
            reward = self.world.evaluate_action(action)
            self.receiver.learn_from_feedback(reward)
            self.sender.learn_from_feedback(reward)
            eval_rewards += reward
            epoch+=1

            if epoch % eval_size == 0:
                print(f'Epoch {epoch}, last {eval_size} epochs reward: {eval_rewards/eval_size}')
                print(world_state, message, action, reward)
                if abs(1-eval_rewards/eval_size) > threshold:
                    eval_rewards = 0
                else:
                    break

        print("Observation to message mapping:")
        print(self.sender.message_weights.argmax(1))
        print("Message to action mapping:")
        print(self.receiver.action_weights.argmax(1))
        