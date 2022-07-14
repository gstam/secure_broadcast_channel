class Queue():
    def __init__(self, capacity):
        self.backlog = 0
        self.capacity = capacity

    def get_backlog(self):
        return self.backlog
    
    def get_capacity(self):
        return self.capacity
    
    def set_backlog(self, backlog):
        self.backlog = backlog
    
    def set_capacity(self, capacity):
        self.capacity = capacity
    
    def packet_arrival(self):
        if self.backlog < self.capacity:
            self.backlog = self.backlog + 1

    def packet_departure(self):
        self.backlog = self.backlog - 1
        
    def reset(self):
        self.backlog = 0