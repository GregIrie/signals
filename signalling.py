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
    def __init__(self, n_states: int, seed: int = 1701):
        self.n_states = n_states
        self.state = 0
        self.rng = np.random.RandomState(seed)
        
    def emit_state(self) -> int:
        self.state = self.rng.randint(self.n_states)
        return self.state
    
    def evaluate_action(self, action: int) -> int:
        return 1 if action == self.state else -1