# Ray Tracing Practice
Reinforcement learning is a powerful approach to learn optimal policies for systems. However, most reinforcement learning algorithms would do a few trial-and-error explorations at the beginning of the learning stage. This trial-and-error investigations might violate safety constraints and damage to the systems in the real world. This project implements a practical approach to help the reinforcement learning algorithms to learn the optimal policies safely. The autonomous drive system is used for demo.

## Reinforcement Learning with the Safety Layer
The project uses the autonomous drive system for a demo. The practical reinforcement learning model is combined with a neural network based safety layer. The safety layer plays a constraining role to restrict the learned parameters not causing safety violations (vehicle driving off the roads). 
![Alt text](./images/safety-layer.png?raw=true "safetyLayer")

The effects of the safety layer could be revealed by the following scenes. In the model without the safety layer, the vehicle tends to drive off the road at the very early stage of learning. However, in the model with the safety layer, the vehicle would be pushed back to the road if it tends to violate the safety constraints.
![Alt text](./images/image14.gif?raw=true "Off")
![Alt text](./images/image15.gif?raw=true "Push")

The details of the project could be found from the report and the presentation. 

## Reference
[Learning to drive in a day](https://arxiv.org/pdf/1807.00412.pdf)
[Self driving car sandbox](https://github.com/tawnkramer/sdsandbox)
[Learning to drive smoothly in minutes](https://github.com/araffin/learning-to-drive-in-5-minutes/)
[Safe exploration in continuous action spaces](https://arxiv.org/pdf/1801.08757.pdf)
