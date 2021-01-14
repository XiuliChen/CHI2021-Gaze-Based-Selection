# CHI2021-Gaze-Based-Selection
We built the gaze-based selection task as an OpenAI gym like environment (see folder 'envs') and used the Stable Baselines3's implementation of the deep RL algorithm, Proximal Policy
Optimization to train the model\footnote{\url{https://github.com/DLR-RM/stable-baselines3}}. Hyperparameters were as follows: Horizon=$500$, Clipping:=$0.15$, Gamma=$0.99$. Other hyperparameters for the model are defaults as in Stable Baselines3 implementation. The hyperparameters for the human bounds used in the model is provided in Table 1.



