import torch
import torch.nn as nn


def get_grad(actions):
    """
    Args:
        actions:
            (从vima中获得的字典,注意不是env.action_space)
            {
                "pose0_position": [ , ], "pose0_rotation": [ , , , ],
                "pose1_position": [ , ], "pose1_rotation": [ , , , ]
            }
    """
    # actions:从vima中获得的字典,注意不是env.action_space
    output = []
    output.extend(actions["pose0_position"])
    output.extend(actions["pose0_rotation"])
    output.extend(actions["pose1_position"])
    output.extend(actions["pose1_rotation"])
    output = torch.Tensor(output)
    targets = torch.zeros([12], requires_grad=True)
    loss = nn.MSELoss()(output, targets)
    loss.backward()
    print(loss.item())


actions = {
    "pose0_position": [0.4099999964237213, -0.15000000596046448],
    "pose0_rotation": [0.0, 0.0, 0.0, 0.9600000381469727],
    "pose1_position": [0.3700000047683716, 0.36000001430511475],
    "pose1_rotation": [0.0, 0.0, 0.9600000381469727, -0.24000000953674316]
}
get_grad(actions)