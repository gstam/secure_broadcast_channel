class BernoulliProcess():
    
    def init(self, success_probability):
        self.success_probability = success_probability
    
    def set_success_probability(self, success_probability):
        self.success_probability = success_probability
        
    def get_success_probability(self):
        return self.success_probability
    
    def generate_event(self):
        if np.random.default_rng().uniform(0., 1., 1) < self.success_probability:
            success = True
        else:
            success = False
        return success 


