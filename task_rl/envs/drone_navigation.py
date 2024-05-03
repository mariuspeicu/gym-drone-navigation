import gym
from gym import spaces
import pygame
import numpy as np


class DroneNavigation(gym.Env):
    WINDOW_SIZE=512
    RENDER_FPS=4
    GRID_SIZE=4
    START_POSITION=(0, 0)
    MAX_STEPS =100
    ZERO=0
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": RENDER_FPS}

    def __init__(self, render_mode=None, size=GRID_SIZE, start_position=START_POSITION, max_steps=MAX_STEPS):
        super(DroneNavigation, self).__init__()
        self.size = size  # The size of the square grid
        self.window_size = self.WINDOW_SIZE  # The size of the PyGame window
        self.start_position = np.array(start_position) # Drone's start positons
        self.max_steps = max_steps # Maximum number of steps the drone can do
        self.current_step = self.ZERO # Current step initialized to 0 

        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )       

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)
        
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and prepares for a new episode. This method is
        called at the start of each new episode. It sets the agent's start position, randomly
        selects a new target location, and ensures that the target is not at the agent's initial position.

        Parameters:
            seed (int, optional): An optional random seed for reproducibility of the environment's initialization.
            options (dict, optional): Additional options that could be passed for environment setup.

        Returns:
            dict: The initial observation of the environment after reset, which includes the current
                positions of the agent and the target.

        Note:
            If the rendering mode is set to 'human', the initial state will be rendered on the screen.
        """
        self.current_step = 0

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose Agent's start position. During the first reset the position will be the one passed in constructor 
        self._agent_location = self.start_position 
                
        #Initialize the target's location randomly 
        self._initialize_target()

        # Set a random position to be used later 
        self.start_position = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation
        
    def step(self, action):
        """
        Executes a single step in the environment by applying an action chosen by the agent. This method updates the
        agent's position based on the action and calculates the reward based on the change in distance to the target.

        Parameters:
            action (int): An integer representing the action to be taken by the agent. The action corresponds to moving
                        in one of four possible directions defined in `_action_to_direction`.

        Returns:
            tuple:
                - dict: The new observation of the environment after the action has been applied. This includes the updated
                        positions of the agent and the target.
                - float: The reward received after executing the action.
                - bool: A boolean flag indicating whether the episode has terminated. An episode terminates if the agent
                        reaches the target or if the maximum number of steps (`max_steps`) has been reached.
                - dict: Additional information about the current state, such as the distance to the target.

        Notes:
            The agent's movement is constrained within the bounds of the environment grid by using `np.clip` to ensure
            the agent does not move outside the defined area. If the rendering mode is set to 'human', the environment
            will also be visually rendered after the step is executed.
        """
        #Save the previous location
        self._previous_location = self._agent_location

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self._move_agent(self._action_to_direction[action])
        
        # An episode is done iff the agent has reached the target or if all the available steps were executed
        terminated, reward = self._reward_function()
        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True
    
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, info
    
    
    def render(self, mode):
        """
        Renders the environment based on the specified mode. This method sets the rendering mode and
        displays the current state of the environment visually.

        Parameters:
            mode (str): The rendering mode to be used. Valid modes include 'human' for on-screen rendering
                        and 'rgb_array' for returning an image of the environment as an array.

        Notes:
            This method updates the `render_mode` attribute of the environment and calls `_render_frame()`
            to carry out the rendering process. Care should be taken when changing rendering modes frequently
            as it can affect performance and behavior of the rendering.
        """
        self.render_mode = mode
        self._render_frame()

    def _get_obs(self):
        """
        Retrieves the current observation of the environment for internal use. This method
        provides a snapshot of the agent's and target's positions in the environment
        Returns:
            dict: A dictionary containing:
                - 'agent': An array representing the current location of the agent in the environment.
                - 'target': An array representing the location of the target the agent is supposed to reach.
        """
        return {"agent": self._agent_location, "target": self._target_location}
    

    def _get_info(self):
        """
        Retrieves additional information about the current state of the environment. This method
        is intended for internal use and provides metrics that could be useful for analysis or
        debugging during the simulation.

        Returns:
            dict: A dictionary containing calculated metrics, including:
                - 'distance': The Manhattan (L1 norm) distance between the agent's and the target's
                current positions.
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def _initialize_target(self):
        """
        Initializes the target's location within the environment grid. This method randomly selects a
        new position for the target, ensuring it does not overlap with the agent's current location.
        
        This is an internal method intended to be used during environment setup or reset, to place
        the target in a valid starting position.

        Notes:
            The method uses a while loop to continuously sample the grid until a suitable location
            that does not coincide with the agent's current position is found. This ensures that
            the agent does not start the episode already at the target location.
        """
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )
    def _move_agent(self, direction):
        """
        Updates the agent's position on the grid based on the specified direction. This method applies
        the given direction vector to the agent's current position and ensures that the new position
        does not exceed the boundaries of the environment grid.

        Parameters:
            direction (numpy.ndarray): A vector indicating the direction and magnitude of the movement. 
                                    The direction is typically a unit vector.

        Notes:
            The agent's position is constrained within the grid using `numpy.clip` to prevent it from 
            moving outside the defined environment size. This method is designed for internal use within 
            the class to handle movement logic efficiently and safely.
        """
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
    
    def _reward_function(self):
        """
        Calculates the reward based on the change in distance between the agent's current position and
        the target's location. This method encourages the agent to move closer to the target by 
        providing a higher reward for reduced distance, and penalizing increased or unchanged distance.

        The function also determines whether the agent has reached the target, which terminates the episode.

        Returns:
            tuple:
                - bool: True if the agent has reached the target (i.e., current distance is zero), 
                        otherwise False.
                - float: The calculated reward for the current step. The reward is increased by 100 
                        if the agent reaches the target, decreased by 1 if the distance to the target 
                        has increased or remained the same, and adjusted by the difference in distance 
                        otherwise.

        Notes:
            - This function uses the Manhattan distance (L1 norm) to calculate the distances.
            - The agent's position before the current action is taken into account when calculating
            the previous distance.
        """
        current_distance = np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        previous_distance = np.linalg.norm(
                self._previous_location - self._target_location, ord=1
            )
        
        reward = previous_distance - current_distance        
        if current_distance == 0:
            reward += 100
            return True, reward

        if current_distance >= previous_distance:
            reward -= 1
        
        return False, reward

    def _render_frame(self): 
        """
        Renders the current state of the environment onto the screen or as an RGB array, depending on the
        set render_mode. This method is responsible for drawing the environment's grid, the agent, and the
        target using Pygame, and updating the display appropriately.

        This method supports two rendering modes:
        - 'human': Renders the environment to the screen and ensures it updates at a specified frame rate.
        - 'rgb_array': Returns an RGB array representation of the environment suitable for recording or further processing.

        Notes:
            - If the window or the clock has not been initialized, this method will initialize them upon the first call
            when 'human' mode is active.
            - The environment grid, target, and agent are visually represented within a Pygame window. Grid lines are added
            for visual clarity.
        """ 
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """
        Properly closes the rendering window and shuts down Pygame, ensuring that all resources are
        correctly released. This method should be called when the environment is no longer needed,
        especially when using the 'human' rendering mode to avoid leaving Pygame windows open.

        Notes:
            This method checks if a Pygame window has been initialized (i.e., `self.window` is not None).
            If a window exists, it closes the window and terminates the Pygame instance to free up system resources.
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
