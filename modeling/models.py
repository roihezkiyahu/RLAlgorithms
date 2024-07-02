import torch.nn as nn
import torch.nn.functional as F
import torch


class CNNDQNAgent(nn.Module):
    def __init__(self, input_shape, output_size,
                 conv_layers_params=[
                     {'in_channels': 11, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                     {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1},
                     {'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1}
                 ],
                 fc_layers=[256], dueling=False, reconnect_extra_features=False):
        super(CNNDQNAgent, self).__init__()
        self.dueling = dueling
        self.conv_layers = nn.ModuleList([nn.Conv2d(**params) for params in conv_layers_params])
        self.fc_input_dim = self._feature_size(input_shape) + int(reconnect_extra_features)*(input_shape[0]-1)

        if dueling:
            self.value_layers = nn.ModuleList()
            self.advantage_layers = nn.ModuleList()

            for i, units in enumerate(fc_layers):
                in_features = self.fc_input_dim if i == 0 else fc_layers[i - 1]
                self.value_layers.append(nn.Linear(in_features, units))
                self.advantage_layers.append(nn.Linear(in_features, units))

            self.value_output = nn.Linear(fc_layers[-1], 1)
            self.advantage_output = nn.Linear(fc_layers[-1], output_size)

        else:
            self.fc_layers = nn.ModuleList()
            for i, units in enumerate(fc_layers):
                in_features = self.fc_input_dim if i == 0 else fc_layers[i - 1]
                self.fc_layers.append(nn.Linear(in_features, units))
            self.fc_layers.append(nn.Linear(fc_layers[-1], output_size))
        self.reconnect_extra_features = reconnect_extra_features

    def forward(self, x):
        if self.reconnect_extra_features:
            x_extra = x[:,1:,:,:]
            x_extra = nn.AdaptiveAvgPool2d(1)(x_extra)
            x_extra = x_extra.view(x_extra.size(0),-1)
        x = self._apply_layers(self.conv_layers, x)
        x = x.view(x.size(0), -1)  # Flatten
        if self.reconnect_extra_features:
            x = torch.cat([x, x_extra],1)

        if self.dueling:
            value = self._apply_layers(self.value_layers, x)
            value = self.value_output(value)
            advantages = self._apply_layers(self.advantage_layers, x)
            advantages = self.advantage_output(advantages)
            q_vals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_vals = self._apply_layers(self.fc_layers, x, True)
        return q_vals

    def _apply_layers(self, layers, x, dont_aplly_last=False):
        for i, layer in enumerate(layers):
            if dont_aplly_last and i == len(layers) - 1:
                return layer(x)
            x = F.relu(layer(x))
        return x

    def _feature_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self._apply_layers(self.conv_layers, x)
            return x.view(1, -1).size(1)


class DQN(nn.Module):
    def __init__(self, layer_params, n_observations, n_actions, dueling=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        if dueling:
            self.value_layers, self.advantage_layers = self._build_dueling_layers(layer_params, n_observations)
            self.value_output = nn.Linear(layer_params[-1]['out_features'], 1)
            self.advantage_output = nn.Linear(layer_params[-1]['out_features'], n_actions)
        else:
            self.layers = self._build_layers(layer_params, n_observations)
            self.output_layer = nn.Linear(layer_params[-1]['out_features'], n_actions)

    def dueling_forward(self, x):
        value = self._apply_layers(self.value_layers, x)
        value = self.value_output(value)
        advantages = self._apply_layers(self.advantage_layers, x)
        advantages = self.advantage_output(advantages)
        q_vals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

    def forward(self, x):
        if self.dueling:
            q_vals = self.dueling_forward(x)
        else:
            q_vals = self._apply_layers(self.layers, x)
            q_vals = self.output_layer(q_vals)
        return q_vals

    @staticmethod
    def _build_dueling_layers(layer_params, input_size):
        value_layers = nn.ModuleList()
        advantage_layers = nn.ModuleList()
        for params in layer_params:
            output_size = params['out_features']
            value_layers.append(nn.Linear(input_size, output_size))
            advantage_layers.append(nn.Linear(input_size, output_size))
            input_size = output_size
        return value_layers, advantage_layers

    @staticmethod
    def _build_layers(layer_params, input_size):
        layers = nn.ModuleList()
        for params in layer_params:
            output_size = params['out_features']
            layers.append(nn.Linear(input_size, output_size))
            input_size = output_size
        return layers

    @staticmethod
    def _apply_layers(layers, x, apply_last=False):
        for i, layer in enumerate(layers):
            if apply_last and i == len(layers) - 1:
                return layer(x)
            x = F.relu(layer(x))
        return x

    @staticmethod
    def _compute_dueling_q_vals(value, advantages):
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class ActorCritic(nn.Module):
    def __init__(self, input_shape, action_size, conv_layers_params, fc_layers, mode='actor', use_instance_norm=False):
        super(ActorCritic, self).__init__()

        self.mode = mode
        self.use_instance_norm = use_instance_norm
        self.conv_layers = nn.ModuleList()
        for params in conv_layers_params:
            self.conv_layers.append(nn.Conv2d(**params))
            if self.use_instance_norm:
                self.conv_layers.append(nn.InstanceNorm2d(params['out_channels']))

        self.fc_input_dim = self._feature_size(input_shape)

        self.fc_layers = nn.ModuleList()
        for i, units in enumerate(fc_layers):
            in_features = self.fc_input_dim if i == 0 else fc_layers[i - 1]
            self.fc_layers.append(nn.Linear(in_features, units))
            if self.use_instance_norm:
                self.fc_layers.append(nn.InstanceNorm1d(units))

        if mode == 'actor':
            self.output_layer = nn.Linear(fc_layers[-1], action_size)
        else:
            self.output_layer = nn.Linear(fc_layers[-1], 1)

    def forward(self, x):
        x = self._apply_layers(self.conv_layers, x)
        x = x.view(x.size(0), -1)
        x = self._apply_layers(self.fc_layers, x)
        x = self.output_layer(x)

        if self.mode == 'actor':
            return x
        else:
            return x

    def _apply_layers(self, layers, x):
        for layer in layers:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                x = F.relu(layer(x))
            elif isinstance(layer, (nn.InstanceNorm2d, nn.InstanceNorm1d)):
                x = layer(x)
        return x

    def _feature_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self._apply_layers(self.conv_layers, x)
            return x.view(1, -1).size(1)
