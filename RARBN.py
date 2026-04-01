import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers, models, Input, optimizers
from tensorflow.keras.layers import Concatenate
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

tfd = tfp.distributions
tfpl = tfp.layers


class RARBN:
    def __init__(self,
                 dim_x: int,
                 num_train: int,
                 enable_ra: bool = False,
                 target_safety: float = 0.95,
                 lr_net: float = 0.005,
                 debate_strength: float = 1.0,
                 n_clusters: int = 10,
                 tube_k: float = 1.0,
                 max_lambda: float = 100.0,
                 id_threshold: float = 2.0):

        self.dim_x = dim_x
        self.num_train = num_train
        self.enable_ra = enable_ra
        self.target_safety = target_safety
        self.debate_strength = debate_strength
        self.opt_net = optimizers.Adam(learning_rate=lr_net)

        # Variables for Elastic Tube & Constraints
        self.tube_k = tf.Variable(tube_k, trainable=False, dtype=tf.float32)
        self.max_lambda = tf.Variable(max_lambda, trainable=False, dtype=tf.float32)
        self.id_threshold = tf.Variable(id_threshold, trainable=False, dtype=tf.float32)

        # K-Means geometric judge parameters
        self.n_clusters = n_clusters
        self.centroids = tf.Variable(tf.zeros([n_clusters, dim_x]), trainable=False, dtype=tf.float32)
        self.cluster_radii = tf.Variable(tf.zeros([n_clusters]), trainable=False, dtype=tf.float32)
        self.residual_std = tf.Variable(0.1, trainable=False, dtype=tf.float32)
        self.kmeans_fitted = False
        self.early_stop_tol = 1e-3

        if self.enable_ra:
            self.log_lambda = tf.Variable(0.0, trainable=True, dtype=tf.float32, name="log_lambda")
            self.lr_lambda = 0.05

        # Build the core BNN model
        self.model = self._build_network()

    def _build_network(self) -> models.Model:
        dim_lf = 1
        kl_scale = 1e-6
        kl_var = tf.Variable(kl_scale, trainable=False, dtype=tf.float32)

        # Scale KL divergence by the number of training samples
        kl_fn = lambda q, p, _: (tfd.kl_divergence(q, p) / self.num_train) * kl_var

        input_x = Input(shape=(self.dim_x,), name="In_X")
        input_lf = Input(shape=(dim_lf,), name="In_LF")

        x_feat = layers.Dense(64, activation='relu')(input_x)
        lf_feat = layers.Dense(16, activation='tanh')(input_lf)
        merged_input = Concatenate()([x_feat, lf_feat])

        h1 = layers.Dense(64, activation='relu')(merged_input)

        def residual_block(xi, units):
            fo = tfpl.DenseFlipout(units, kernel_divergence_fn=kl_fn, activation='relu')(xi)
            if xi.shape[-1] == units:
                return layers.Add()([xi, fo])
            else:
                return fo

        h2 = residual_block(h1, 64)
        h3 = layers.Dense(32, activation='relu')(h2)
        feature_layer = residual_block(h3, 32)

        delta_params = tfpl.DenseFlipout(2, kernel_divergence_fn=kl_fn, name="Head_Delta")(feature_layer)
        rho_raw = layers.Dense(1, activation='tanh', name="Head_Rho_Raw")(feature_layer)
        rho_out = layers.Lambda(lambda x: (x + 1.0), name="Head_Rho")(rho_raw)

        combined_out = Concatenate()([delta_params, rho_out])

        model = models.Model(inputs=[input_x, input_lf], outputs=combined_out, name="RARBN_Core")
        return model

    def _fit_kmeans_judge(self, X_tr: np.ndarray, yl_tr: np.ndarray, yh_tr: np.ndarray):
        # Fit K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(X_tr)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Calculate minimum radius to prevent singular clusters
        if len(X_tr) > 100:
            sample_idx = np.random.choice(len(X_tr), 100, replace=False)
            sample_X = X_tr[sample_idx]
        else:
            sample_X = X_tr

        dists_matrix = cdist(sample_X, sample_X)
        np.fill_diagonal(dists_matrix, np.inf)
        avg_spacing = np.mean(np.min(dists_matrix, axis=1))
        min_radius = avg_spacing * 2.5

        radii = []
        for i in range(self.n_clusters):
            cluster_points = X_tr[labels == i]
            if len(cluster_points) == 0:
                radii.append(min_radius)
                continue
            dists = np.linalg.norm(cluster_points - centers[i], axis=1)
            raw_r = np.percentile(dists, 95.0)
            radii.append(max(raw_r, min_radius))

        self.centroids.assign(tf.constant(centers, dtype=tf.float32))
        self.cluster_radii.assign(tf.constant(radii, dtype=tf.float32))

        # Calculate residual margin based on LF vs HF differences
        residuals = yh_tr - yl_tr
        res_std = np.std(residuals)
        res_std = max(res_std, 0.05)  # Provide minimum margin to avoid overfitting
        self.residual_std.assign(tf.constant(res_std, dtype=tf.float32))

        self.kmeans_fitted = True

    def fit(self, X_tr: np.ndarray, yl_tr: np.ndarray, yh_tr: np.ndarray, epochs: int = 2000,
            patience: int = 200) -> list:
        d_tr = [tf.constant(X_tr, dtype=tf.float32), tf.constant(yl_tr.reshape(-1, 1), dtype=tf.float32)]
        y_tr = tf.constant(yh_tr.reshape(-1, 1), dtype=tf.float32)

        history_rmse = []
        best_rmse = float('inf')
        wait = 0

        @tf.function
        def train_step_ra():
            with tf.GradientTape() as tape:
                raw_out = self.model(d_tr, training=True)
                delta_mu, delta_logstd, rho = raw_out[..., 0:1], raw_out[..., 1:2], raw_out[..., 2:3]

                sigma = 1e-3 + tf.math.softplus(delta_logstd)
                recon_mu = (rho * d_tr[1]) + delta_mu
                safe_sigma = tf.maximum(sigma, 1e-6)

                nll = tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * safe_sigma ** 2) +
                                     ((y_tr - recon_mu) ** 2) / (2 * safe_sigma ** 2))
                diff = y_tr - recon_mu
                soft_is_safe = tf.nn.sigmoid(diff / 0.05)
                current_safe_ratio = tf.reduce_mean(soft_is_safe)
                violation = self.target_safety - current_safe_ratio

                current_lambda = tf.clip_by_value(tf.math.exp(self.log_lambda), 0.01, self.max_lambda)
                penalty = tf.where(diff < 0, tf.square(diff), tf.square(diff))
                loss_asym = tf.reduce_mean(penalty)
                rho_reg = 0.01 * tf.reduce_mean(tf.square(rho - 1.0))

                total_loss = nll + sum(self.model.losses) + (0.1 * tf.reduce_mean(sigma)) + \
                             (current_lambda * violation) + loss_asym + rho_reg

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.opt_net.apply_gradients(zip(grads, self.model.trainable_variables))
            self.log_lambda.assign_add(self.lr_lambda * violation)
            return total_loss

        @tf.function
        def train_step_standard():
            with tf.GradientTape() as tape:
                raw_out = self.model(d_tr, training=True)
                delta_mu, delta_logstd, rho = raw_out[..., 0:1], raw_out[..., 1:2], raw_out[..., 2:3]

                sigma = 1e-3 + tf.math.softplus(delta_logstd)
                recon_mu = (rho * d_tr[1]) + delta_mu
                safe_sigma = tf.maximum(sigma, 1e-6)

                nll = tf.reduce_mean(0.5 * tf.math.log(2 * np.pi * safe_sigma ** 2) +
                                     ((y_tr - recon_mu) ** 2) / (2 * safe_sigma ** 2))
                rho_reg = 0.01 * tf.reduce_mean(tf.square(rho - 1.0))

                total_loss = nll + sum(self.model.losses) + rho_reg

            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.opt_net.apply_gradients(zip(grads, self.model.trainable_variables))
            return total_loss

        @tf.function
        def calculate_rmse():
            raw_out = self.model(d_tr, training=False)
            delta_mu, rho = raw_out[..., 0:1], raw_out[..., 2:3]
            recon_mu = (rho * d_tr[1]) + delta_mu
            return tf.sqrt(tf.reduce_mean(tf.square(y_tr - recon_mu)))

        for ep in range(epochs):
            if self.enable_ra:
                train_step_ra()
            else:
                train_step_standard()

            curr_rmse = float(calculate_rmse().numpy())
            history_rmse.append(curr_rmse)

            # Early stopping check
            if curr_rmse < best_rmse - 1e-5:
                best_rmse = curr_rmse
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

        # Fit K-Means judge at the end of training using the residuals
        self._fit_kmeans_judge(X_tr, yl_tr, yh_tr)
        return history_rmse

    @tf.function
    def _compute_kmeans_weight(self, x_input: tf.Tensor):
        """Computes out-of-distribution physical weights based on K-Means centroids."""
        diff = tf.expand_dims(x_input, 1) - tf.expand_dims(self.centroids, 0)
        dists_sq = tf.reduce_sum(tf.square(diff), axis=2)
        dists = tf.sqrt(tf.maximum(dists_sq, 1e-9))

        min_dists = tf.reduce_min(dists, axis=1)
        nearest_idx = tf.argmin(dists, axis=1)
        nearest_radii = tf.gather(self.cluster_radii, nearest_idx)

        ratio = min_dists / (nearest_radii + 1e-6)
        raw_excess = ratio - self.id_threshold

        activation = tf.nn.relu(raw_excess)
        w_phys = tf.clip_by_value(activation, 0.0, 50.0)

        return tf.reshape(w_phys, [-1, 1])

    def _debate_optimization(self, delta_init, mean_nn_delta, rho_val, y_lf_val, x_input):
        delta_opt = delta_init
        m = tf.zeros_like(delta_init)
        v = tf.zeros_like(delta_init)
        lr = 0.01
        beta1, beta2 = 0.9, 0.999

        w_phys = self._compute_kmeans_weight(x_input)
        adaptive_strength = self.debate_strength * w_phys

        # Elastic bounds based on tube_k and residual std
        margin_upper = self.tube_k * self.residual_std
        margin_lower = self.tube_k * self.residual_std

        for t in range(1, 201):
            with tf.GradientTape() as tape:
                tape.watch(delta_opt)

                # Data Loss: Anchor to NN initial guess
                e_data = tf.reduce_mean(tf.square(delta_opt - mean_nn_delta))

                # Physics Loss: Elastic Tube constraints
                y_pred_curr = (rho_val * y_lf_val) + delta_opt
                diff = y_pred_curr - y_lf_val

                loss_upper = tf.square(tf.nn.relu(diff - margin_upper))
                loss_lower = tf.square(tf.nn.relu(-diff - margin_lower))

                phys_loss = tf.reduce_mean(adaptive_strength * (loss_upper + loss_lower))
                total_energy = e_data + phys_loss

            grads = tape.gradient(total_energy, delta_opt)

            if self.early_stop_tol > 0.0:
                max_grad = tf.reduce_max(tf.abs(grads))
                if max_grad < self.early_stop_tol:
                    break

            # Adam optimization steps
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * tf.square(grads)
            m_hat = m / (1 - beta1 ** tf.cast(t, tf.float32))
            v_hat = v / (1 - beta2 ** tf.cast(t, tf.float32))
            delta_opt = delta_opt - lr * m_hat / (tf.sqrt(v_hat) + 1e-7)

        return delta_opt

    def predict(self, X: np.ndarray, yl: np.ndarray, n_mc: int = 50) -> tuple:
        d_in = [tf.constant(X, dtype=tf.float32), tf.constant(yl.reshape(-1, 1), dtype=tf.float32)]

        if n_mc > 1:
            all_delta_mu, all_rho, all_sigma_delta = [], [], []
            for _ in range(n_mc):
                raw_out = self.model(d_in, training=True)
                all_delta_mu.append(raw_out[..., 0:1])
                all_rho.append(raw_out[..., 2:3])
                all_sigma_delta.append(1e-3 + tf.math.softplus(raw_out[..., 1:2]))

            delta_mu = tf.reduce_mean(tf.stack(all_delta_mu), axis=0)
            rho = tf.reduce_mean(tf.stack(all_rho), axis=0)
            sigma_delta = tf.reduce_mean(tf.stack(all_sigma_delta), axis=0)
        else:
            raw_out = self.model(d_in, training=False)
            delta_mu = raw_out[..., 0:1]
            rho = raw_out[..., 2:3]
            sigma_delta = 1e-3 + tf.math.softplus(raw_out[..., 1:2])

        delta_final = self._debate_optimization(
            delta_init=delta_mu,
            mean_nn_delta=delta_mu,
            rho_val=rho,
            y_lf_val=d_in[1],
            x_input=d_in[0]
        )

        y_final = (rho * d_in[1]) + delta_final
        return y_final.numpy().flatten(), sigma_delta.numpy().flatten()