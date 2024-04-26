class R_NVP(nn.Module):
    def __init__(self, d, k, hidden, base_dist, shared=False, init_identity=False, even_odd=False):
        super().__init__()
        self.d, self.k = d, k
        self.base_dist = base_dist
        self.shared = shared
        self.even_odd = even_odd
        if self.shared :
          self.shared_net = nn.Sequential(
                    nn.Linear(k, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, 2*(d - k)))
          if init_identity :
            nn.init.zeros_(self.shared_net[2].weight)
            nn.init.zeros_(self.shared_net[2].bias)
        else :
          self.sig_net = nn.Sequential(
                    nn.Linear(k, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, d - k))
          self.mu_net = nn.Sequential(
                    nn.Linear(k, hidden),
                    nn.LeakyReLU(),
                    nn.Linear(hidden, d - k))
          if init_identity :
            nn.init.zeros_(self.sig_net[2].weight)
            nn.init.zeros_(self.sig_net[2].bias)
            nn.init.zeros_(self.mu_net[2].weight)
            nn.init.zeros_(self.mu_net[2].bias)

    def forward(self, x, flip=False):
        #print(x.shape)
        if not self.even_odd:
          x1, x2 = x[:, :self.k], x[:, self.k:]
        else:
          x1, x2 = x[:, ::2], x[:, 1::2]

        if flip:
            x2, x1 = x1, x2

        # forward
        #print(x1.shape)
        if self.shared :
          shared_output = self.shared_net(x1)
          #shared_output.requires_grad_()
          sig, mu = shared_output[:,:d-k], shared_output[:,d-k:]
          z1, z2 = x1, x2 * torch.exp(sig) + mu
        else :
          sig = self.sig_net(x1)
          mu = self.mu_net(x1)
          z1, z2 = x1, x2 * torch.exp(sig) + mu

        if flip:
            z2, z1 = z1, z2
        #print(z1.shape, z2.shape)
        z_hat = torch.cat([z1, z2], dim=-1)
        log_jacob = sig.sum(-1)

        return z_hat, log_jacob

    def inverse(self, Z, flip=False):
        z1, z2 = Z[:, :self.k], Z[:, self.k:]

        if flip:
            z2, z1 = z1, z2

        x1 = z1
        if self.shared :
          shared_output = self.shared_net(z1)
          sig, mu = shared_output[:,:d-k], shared_output[:,d-k:]
          #print(z2.shape, sig.shape, mu.shape)
          x2 = (z2 - mu) * torch.exp(-sig)
        else :
          x2 = (z2 - self.mu_net(z1)) * torch.exp(-self.sig_net(z1))

        if flip:
            x2, x1 = x1, x2
        if not self.even_odd:
          return torch.cat([x1, x2], -1)
        else:
          temp1, temp2 = torch.zeros_like(torch.cat([x1, x2], -1)), torch.zeros_like(torch.cat([x1, x2], -1))
          temp1[:,::2] = x1
          temp2[:,1::2] = x2
          return temp1 + temp2

class stacked_NVP(nn.Module):
    def __init__(self, d, k, hidden, n, base_dist, shared=False, init_identity=False, even_odd=False):
        super().__init__()
        self.bijectors = nn.ModuleList([
            R_NVP(d, k, hidden=hidden, base_dist=base_dist, shared=shared, init_identity=init_identity, even_odd=even_odd) for _ in range(n)
        ])
        self.flips = [True if i%2 else False for i in range(n)]

    def forward(self, x):
        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):
            x, lj = bijector(x, flip=self.flips[f])
            log_jacobs.append(lj)

        return x, sum(log_jacobs)

    def inverse(self, z):
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            z = bijector.inverse(z, flip=self.flips[f])
        return z

# Implementation with a Real NVP flow model
class AugmentedInvertibleKoopmanVAE(nn.Module):
    def __init__(self, input_dim:int, hidden_dim=64, n_layers_encoder=3, augmentation_dims=[256,128, 16], even_odd=False, random_K=False, positive_nonlin=torch.abs, device='cpu'):
        """
        Koopman Autoencoder class, comprising an auto-encoder and a Koopman matrix.

        Args:
            input_dim (int): Dimension of the input data.
            linear_dims (list): List of linear layer dimensions.
            device (str, optional): Device to run the model on (default: 'cpu').
        """
        super(AugmentedInvertibleKoopmanVAE, self).__init__()


        # Encoder
        self.input_dim = input_dim
        self.augmentation_dim = augmentation_dims[-1]
        self.latent_dim = input_dim + augmentation_dims[-1]
        self.positive_nonlin = positive_nonlin
        base_mu, base_cov = torch.zeros(self.latent_dim), torch.eye(self.latent_dim)
        base_dist = MultivariateNormal(base_mu, base_cov)
        self.invertible_encoder = stacked_NVP(input_dim, input_dim // 2, hidden=hidden_dim, n=n_layers_encoder, base_dist=base_dist, even_odd=even_odd).to(device)
        self.augmentation_encoder = nn.ModuleList()
        self.augmentation_encoder.add_module("encoder_1", nn.Linear(input_dim, augmentation_dims[0]))
        for i in range(len(augmentation_dims)-2):
            self.augmentation_encoder.add_module(f"encoder_{i+2}", nn.Linear(augmentation_dims[i], augmentation_dims[i+1]))
        self.augmentation_encoder.add_module(f"encoder_{len(augmentation_dims)}", nn.Linear(augmentation_dims[len(augmentation_dims)-2], input_dim + 2*augmentation_dims[-1]))

        # Koopman operator
        if not random_K:
          self.K = torch.eye(self.latent_dim, requires_grad=True, device=device)
        else:
          #self.K = torch.eye(self.latent_dim, requires_grad=True, device=device) + torch.randn((self.latent_dim, self.latent_dim), requires_grad=True, device=device)*1e-6
          #self.K = self.K.clone().detach().requires_grad_()
          M = torch.randn((self.latent_dim, self.latent_dim))
          A = M - M.T
          self.K = torch.matrix_exp(A).to(device).clone().detach().requires_grad_()
          print(self.K.is_leaf)
        self.state_dict()['K'] = self.K

    def encode(self, x):
        """Encode input data x using the encoder layers."""
        mu_invertible, _ = self.invertible_encoder(x)
        augmentation = x
        for layer_idx, layer in enumerate(self.augmentation_encoder):
            augmentation = layer(augmentation)
            if layer_idx < len(self.augmentation_encoder) - 1:
                augmentation = F.relu(augmentation)
        #print(augmentation.shape)
        mu_augmentation = augmentation[:,:model.augmentation_dim]
        #var = torch.abs(augmentation[:,model.augmentation_dim:])
        var = self.positive_nonlin(augmentation[:,model.augmentation_dim:])
        #a = quarantedeux
        return mu_invertible, mu_augmentation, var

    def encode_and_logjacob(self, x):
        """Encode input data x using the encoder layers."""
        mu_invertible, logjacob = self.invertible_encoder(x)
        augmentation = x
        for layer_idx, layer in enumerate(self.augmentation_encoder):
            augmentation = layer(augmentation)
            if layer_idx < len(self.augmentation_encoder) - 1:
                augmentation = F.relu(augmentation)
        #print(augmentation.shape)
        mu_augmentation = augmentation[:,:model.augmentation_dim]
        #var = torch.abs(augmentation[:,model.augmentation_dim:])
        var = self.positive_nonlin(augmentation[:,model.augmentation_dim:])
        #a = quarantedeux
        return mu_invertible, mu_augmentation, var, logjacob

    def log_jacob(self, x):
        _, output = self.invertible_encoder(x)
        return output

    def sample_state(self, mu_invertible, mu_augmentation, var, center=False, scale=1):
      mu = torch.cat([mu_invertible, mu_augmentation], dim=-1)
      if center:
        return mu
      else:
        std = torch.sqrt(var)
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std * scale) # sampling
        return sample

    def sample_state_multiple(self, mu_invertible, mu_augmentation, var, n_samples, fixed_invertible_part=False, scale=1, print_shape=True):
      mu = torch.cat([mu_invertible, mu_augmentation], dim=-1)
      std = torch.sqrt(var)
      if fixed_invertible_part:
        std = std.T
        std[:model.input_dim] = 0
        std = std.T
      eps = torch.randn((n_samples, std.shape[0], std.shape[1])).to(device) # `randn_like` as we need the same size
      if eps.shape[1] == 1:
        eps = eps.squeeze(1)
      if print_shape:
        print(mu.shape, std.shape, eps.shape)
      samples = mu + (eps * std * scale) # sampling
      return samples

    def decode(self, x):
        """Decode the encoded data x using the decoder layers."""
        return self.invertible_encoder.inverse(x[:, :self.input_dim])

    def one_step_ahead(self, x):
        """Predict one-step-ahead in the latent space using the Koopman operator."""
        #return torch.matmul(x, self.K)
        return torch.matmul(self.K, x.T).T

    def one_step_back(self, x):
        """Predict one-step-back in the latent space using the inverse of the Koopman operator."""
        return torch.matmul(x, torch.inverse(self.K))

    def forward(self, x):
        """
        Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input state.

        Returns:
            x_advanced (torch.Tensor): Estimated state after one time step.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded input state advanced by one time step.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_ahead(phi)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def forward_n(self, x, n):
        """
        Perform forward pass for n steps.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Estimated state after n time steps.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced by n time steps.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_ahead(phi)
        for k in range(n-1):
            phi_advanced = self.one_step_ahead(phi_advanced)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def forward_n_remember(self, x, n, training=False, center=False):
        """
        Perform forward pass for n steps while remembering intermediate latent states.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.
            training (bool, optional): Flag to indicate training mode (default: False).

        Returns:
            x_advanced (torch.Tensor or None): Estimated state after n time steps if not training, otherwise None.
            phis (torch.Tensor): Encoded state at each step, concatenated along the 0th dimension.
        """
        mu_invertible, mu_augmentation, var = self.encode(x)
        phis = [self.sample_state(mu_invertible, mu_augmentation, var, center)]
        for k in range(n):
            phis.append(self.one_step_ahead(phis[-1]))
        x_advanced = None if training else self.decode(phis[n])
        return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0)

    def forward_n_multiple(self, x, n, n_samples, scale=1):
      mu_invertible, mu_augmentation, var = self.encode(x)
      phis = [self.sample_state_multiple(mu_invertible, mu_augmentation, var, n_samples, scale=scale, print_shape=False)]
      #print(phis[0].shape)
      for k in range(n):
        if len(phis[-1].shape) == 2:
          phis.append(self.one_step_ahead(phis[-1]))
        else:
          phis_shape = phis[-1].shape
          phis.append(self.one_step_ahead(phis[-1].flatten(0,1)))
          phis[-1] = phis[-1].reshape((phis_shape[0], phis_shape[1], phis_shape[2]))
      #print(phis[n].shape)
      if len(phis[n].shape) == 2:
        x_advanced = self.decode(phis[n])
      else:
        x_advanced = self.decode(phis[n].flatten(0,1)).reshape(phis[n].shape[0], phis[n].shape[1], model.input_dim)
      return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0).transpose(0,1)


    def backward(self, x):
        """
        Perform backward pass through the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            x_advanced (torch.Tensor): Estimated state after one step back.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced one step back.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_back(phi)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def backward_n(self, x, n):
        """
        Perform backward pass for n steps.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Estimated state after n steps back.
            phi (torch.Tensor): Encoded input state.
            phi_advanced (torch.Tensor): Encoded state advanced n steps back.
        """
        phi = self.encode(x)
        phi_advanced = self.one_step_back(phi)
        for k in range(n-1):
            phi_advanced = self.one_step_ahead(phi_advanced)
        x_advanced = self.decode(phi_advanced)
        return x_advanced, phi, phi_advanced

    def backward_n_remember(self, x, n):
        """
        Perform backward pass for n steps while remembering intermediate states.

        Args:
            x (torch.Tensor): Input state.
            n (int): Number of steps to advance.

        Returns:
            x_advanced (torch.Tensor): Reconstructed state after n steps back.
            phis (torch.Tensor): Encoded state at each step, concatenated along the 0th dimension.
        """
        phis = []
        phis.append(self.encode(x))
        for k in range(n):
            phis.append(self.one_step_back(phis[-1]))
        x_advanced = self.decode(phis[n])
        return x_advanced, torch.cat(tuple(phi.unsqueeze(0) for phi in phis), dim=0)

    def sample_likelihood(self, observation, target, delta_t, encoding=False, deterministic_part=False):
      if len(target.shape) == 1:
        target = target.unsqueeze(0)
      if encoding:
        if len(observation.shape) == 1:
          observation = observation.unsqueeze(0)
        mu_invertible, mu_augmentation, var = self.encode(observation)
        mu_obs = torch.cat([mu_invertible, mu_augmentation], dim=-1)
        Sigma = torch.diag(var) if not deterministic_part else torch.diag(var[self.input_dim:])
        mu_invertible_target, mu_augmentation_target, var_target = self.encode(target)
        mu_target = torch.cat([mu_invertible_target, mu_augmentation_target], dim=-1)
      else:
        (mu_obs, var_obs) = observation
        Sigma = var_obs if not deterministic_part else var_obs[self.input_dim:]
        mu_target = target
      mu_delta_t = torch.matmul(torch.matrix_power(self.K, delta_t), mu_obs.T).T
      if not deterministic_part:
        #print(Sigma)
        assert Sigma.shape[1] == self.latent_dim, "when deterministic_part=False, Sigma is expected to be of the size of K."
        left_prod = torch.einsum('ij,bj->bij', torch.matrix_power(self.K, delta_t), Sigma)
        #print(left_prod.shape)
        Sigma_delta_t = torch.matmul(left_prod, torch.matrix_power(self.K.T, delta_t))
      else:
        assert Sigma.shape == (self.augmentation_dim, self.augmentation_dim), "when deterministic_part=True, Sigma is expected to be of the size of the augmentation."
        inner_Sigma = torch.matmul(torch.matmul(self.K[:,self.input_dim:],
                                                Sigma),
                                   self.K[:,self.input_dim:].T)
        Sigma_delta_t = torch.matmul(torch.matmul(torch.matrix_power(self.K, delta_t-1),
                                                  inner_Sigma),
                                     torch.matrix_power(self.K.T, delta_t-1))
      #print(mu_target.shape, mu_delta_t.T.shape, Sigma_delta_t.shape)
      return MGL_several_Sigma(mu_target,
                               mu_delta_t,
                               Sigma_delta_t,
                               log=True)

    def trajectory_likelihood(self, observation, target, delta_t, encoding=False, deterministic_part=False):
      nan = False
      if len(target.shape) <= 2:
        target = target.unsqueeze(0)
      if encoding:
        if len(observation.shape) == 1:
          observation = observation.unsqueeze(0)
        mu_invertible_obs, mu_augmentation_obs, var_obs = self.encode(observation)
        mu_obs = torch.cat([mu_invertible, mu_augmentation], dim=-1)
        Sigma = torch.diag(var) if not deterministic_part else torch.diag(var[self.input_dim:])
        mu_invertible_target, mu_augmentation_target, var_target = self.encode(target)
        mu_target = torch.cat([mu_invertible_target, mu_augmentation_target], dim=-1)
      else:
        (mu_obs, var_obs) = observation
        mu_invertible_obs, mu_augmentation_obs = mu_obs[:,:self.input_dim], mu_obs[:,self.input_dim:]
        Sigma = torch.diag_embed(var_obs) if not deterministic_part else torch.diag_embed(var_obs[self.input_dim:])
        mu_target = target
        #print(mu_target.shape)
        mu_invertible_target, mu_augmentation_target = mu_target[:,:,:self.input_dim], mu_target[:,:,self.input_dim:]
      likelihood = 0
      Sigma += torch.eye(model.latent_dim, device=device)*1e-9
      #print(mu_invertible_target.shape, mu_invertible_obs.shape, Sigma.shape)
      likelihood += MGL_several_Sigma(mu_invertible_target[:,0],
                                      mu_invertible_obs,
                                      Sigma[:,:self.input_dim, :self.input_dim],
                                      log=True, detail=nan)
      #print(torch.isnan(likelihood))
      #print(likelihood.shape)
      if torch.any(torch.isnan(likelihood)):
        print("nan in the initial likelihood (and subsequent ones)")
        nan = True
      #print(delta_t)
      mu_augmentation = mu_augmentation_obs # Initially the covariance is diagonal
      Sigma_augmentation = Sigma if deterministic_part else Sigma[:, self.input_dim:, self.input_dim:]
      for i in range(1, len(delta_t)):
        delta = delta_t[i] - delta_t[i-1]
        z_bar = torch.zeros_like(mu_target[:,i])
        z_bar[:, :self.input_dim] = mu_invertible_target[:,i-1]
        z_bar[:, self.input_dim:] = mu_augmentation
        #print(z_bar.shape)
        K_power = torch.matrix_power(self.K, delta)
        mu_bar = torch.matmul(K_power, z_bar.T).T
        #print(mu_bar.shape)

        Sigma_bar = torch.zeros((Sigma_augmentation.shape[0], self.latent_dim, self.latent_dim)).to(device)
        Sigma_bar[:, self.input_dim:, self.input_dim:] = Sigma_augmentation
        left_prod_Sigma = torch.matmul(K_power, Sigma_bar)
        Sigma_bar = torch.matmul(left_prod_Sigma, torch.matrix_power(self.K.T, delta))
        Sigma_bar += torch.eye(model.latent_dim, device=device) * 1e-9
        Sigma_bar = (Sigma_bar + Sigma_bar.transpose(-1,-2)) / 2
        #print(mu_invertible_target[:, i].shape, mu_bar[:, :self.input_dim].shape, Sigma_bar[:, :self.input_dim, :self.input_dim].shape)
        #print(Sigma_bar[42, :self.input_dim, :self.input_dim])
        #print(f'delta={delta}')
        likelihood += MGL_several_Sigma(mu_invertible_target[:, i],
                                        mu_bar[:, :self.input_dim],
                                        Sigma_bar[:, :self.input_dim, :self.input_dim] + torch.eye(self.input_dim, device=device)*1e-9,
                                        log=True, detail=nan)
        if torch.any(torch.isnan(likelihood)) and not nan:
          print(f"nan after iteration {i} (and subsequent ones)")
          nan = True
        left_prod = torch.matmul(Sigma_bar[:, self.input_dim:, :self.input_dim],
                                 torch.linalg.inv(Sigma_bar[:, :self.input_dim, :self.input_dim]))
        #print(left_prod.shape, (mu_invertible_target[:,i] - mu_bar[:, :self.input_dim]).shape)
        full_prod = torch.matmul(left_prod, (mu_invertible_target[:,i] - mu_bar[:, :self.input_dim]).unsqueeze(-1)).squeeze(-1)
        mu_augmentation = mu_bar[:, self.input_dim:] + full_prod
        #torch.matmul(torch.matmul(Sigma_bar[:, self.input_dim:, :self.input_dim],
                                                                                 #torch.linalg.inv(Sigma_bar[:, :self.input_dim, :self.input_dim])),
                                                                    #mu_invertible_target[:,i] - mu_bar[:, :self.input_dim])
        Sigma_augmentation = Sigma_bar[:, model.input_dim:, model.input_dim:] - torch.matmul(torch.matmul(Sigma_bar[:, self.input_dim:, :self.input_dim],
                                                                                                          torch.linalg.inv(Sigma_bar[:, :self.input_dim, :self.input_dim])),
                                                                                             Sigma_bar[:, :self.input_dim, self.input_dim:])
        Sigma_augmentation = (Sigma_augmentation + Sigma_augmentation.transpose(-1,-2)) / 2
        #a = quarantedeux
      #print(likelihood.shape)
      return likelihood


    def configure_optimizers(self, lr=1e-3, K_lr=None):
        """
        Configure the optimizer for training the model.

        Args:
            lr (float, optional): Learning rate for the optimizer (default: 1e-3).

        Returns:
            torch.optim.Optimizer: Optimizer instance.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        if K_lr is None:
          K_lr = lr
        optimizer.add_param_group({"params": self.K, "lr": K_lr})
        return optimizer
