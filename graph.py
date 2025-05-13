import matplotlib.pyplot as plt

class simDataGraph:

    def __init__(self, config):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.max_line, = self.ax.plot([], [], label="Max Food")
        self.mean_line, = self.ax.plot([], [], label="Mean Food")
        self.cumulative_mean_line, = self.ax.plot([], [], label="Cumulative Mean Food") # New line for cumulative mean
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Food")
        self.ax.set_title("Food stats over Epochs")
        self.ax.legend(loc='upper left') # Adjusted legend location

        self.epochs_max_food = []
        self.epochs_mean_food = []
        self.epochs_cumulative_mean_food = []
        self.epochs_cumulative_mean_derivative = [] # To store derivative
        self.epochs_comulative_mean = 0

        # Initialize text annotations for stats
        # Position them to the right of the plot
        self.stats_text_objects = []
        y_pos_start = 0.85
        y_offset = 0.05
        stats_to_display = [
            "Current Max Food: N/A",
            "Current Mean Food: N/A",
            "Current CuMean: N/A",
            "CuMean Derivative: N/A"
        ]
        for i, text_label in enumerate(stats_to_display):
            text_obj = self.fig.text(0.99, y_pos_start - i * y_offset, text_label, 
                                     horizontalalignment='right', verticalalignment='top', 
                                     transform=self.fig.transFigure, fontsize=8)
            self.stats_text_objects.append(text_obj)
        
        self.fig.subplots_adjust(right=0.75) # Adjust plot to make space for text

    def update(self, current_max,current_mean, epoch):
        self.epochs_max_food.append(current_max)
        self.epochs_mean_food.append(current_mean) 
        
        previous_cumulative_mean = 0
        if self.epochs_cumulative_mean_food: # Check if list is not empty
            previous_cumulative_mean = self.epochs_cumulative_mean_food[-1]

        if not self.epochs_cumulative_mean_food: # For the first epoch
            new_cumulative_mean = current_mean
            self.epochs_cumulative_mean_food.append(new_cumulative_mean)
            self.epochs_cumulative_mean_derivative.append(0) # Derivative is 0 for the first point
        else:
            new_cumulative_mean = (previous_cumulative_mean * epoch + current_mean) / (epoch + 1)
            self.epochs_cumulative_mean_food.append(new_cumulative_mean)
            # Calculate derivative: (current_cumean - previous_cumean)
            # Assuming epoch represents the time step, so dt = 1
            derivative = new_cumulative_mean - previous_cumulative_mean
            self.epochs_cumulative_mean_derivative.append(derivative)

        # update the plot
        self.max_line.set_data(range(len(self.epochs_max_food)), self.epochs_max_food)
        self.mean_line.set_data(range(len(self.epochs_mean_food)), self.epochs_mean_food)
        self.cumulative_mean_line.set_data(range(len(self.epochs_cumulative_mean_food)), self.epochs_cumulative_mean_food)
        
        self.ax.relim()
        self.ax.autoscale_view()

        # Update stats text
        if self.stats_text_objects:
            self.stats_text_objects[0].set_text(f"Current Max Food: {current_max:.2f}")
            self.stats_text_objects[1].set_text(f"Current Mean Food: {current_mean:.2f}")
            current_cumean_val = self.epochs_cumulative_mean_food[-1] if self.epochs_cumulative_mean_food else float('nan')
            self.stats_text_objects[2].set_text(f"Current CuMean: {current_cumean_val:.2f}")
            current_cumean_deriv_val = self.epochs_cumulative_mean_derivative[-1] if self.epochs_cumulative_mean_derivative else float('nan')
            self.stats_text_objects[3].set_text(f"CuMean Derivative: {current_cumean_deriv_val:.3f}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, filename):
        plt.savefig(filename)
    
    def close(self):
        plt.ioff()
        plt.close(self.fig)