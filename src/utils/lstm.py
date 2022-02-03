from src.utils.unrolled_lstm import UnrolledLSTM

class LSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float, batch_first: bool
    ) -> None:
        super().__init__()

        self.lstm = UnrolledLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            batch_first=batch_first,
        )

        self.to_bands = nn.Linear(
            in_features=hparams.forecasting_vector_size, out_features=num_bands
        )

        dataset = self.get_dataset(subset="training", cache=False)
        self.train_count = len(dataset)
        
        self.num_timesteps = dataset.num_timesteps
        self.output_timesteps = self.num_timesteps - self.hparams.input_months

        # we save the normalizing dict because we calculate weighted
        # normalization values based on the datasets we combine.
        # The number of instances per dataset (and therefore the weights) can
        # vary between the train / test / val sets - this ensures the normalizing
        # dict stays constant between them
        self.normalizing_dict: Optional[Dict[str, np.ndarray]] = dataset.normalizing_dict

        self.forecaster_loss = F.smooth_l1_loss

        self.log_recorder = {
            'train': 0,
            'validation': 0,
            'test': 0
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        input_timesteps = x.shape[1]
        assert input_timesteps >= 1

        output = torch.empty((1, 1, 1, 1))
        predicted_output: List[torch.Tensor] = []
        for i in range(input_timesteps):
            # fmt: off
            input = x[:, i: i + 1, :]
            # fmt: on
            output, hidden_tuple = self.lstm(input, hidden_tuple)
            output = self.to_bands(torch.transpose(output[0, :, :, :], 0, 1))
            predicted_output.append(output)

        # we have already predicted the first output timestep (the last
        # output of the loop above)
        for i in range(self.output_timesteps - 1):
            output, hidden_tuple = self.lstm(output, hidden_tuple)
            output = self.to_bands(torch.transpose(output[0, :, :, :], 0, 1))
            predicted_output.append(output)
        return torch.cat(predicted_output, dim=1)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser_args: Dict[str, Tuple[Type, Any]] = {
            "--forecasting_vector_size": (int, 256),
            "--forecasting_dropout": (float, 0.2),
        }

        for key, vals in parser_args.items():
            parser.add_argument(key, type=vals[0], default=vals[1])

        return parser

    def train_dataloader(self):
        return DataLoader(
            self.get_dataset(subset="training"),
            shuffle=True,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.get_dataset(
                subset="validation",
                normalizing_dict=self.normalizing_dict,
            ),
            batch_size=self.hparams.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.get_dataset(
                subset="testing",
                normalizing_dict=self.normalizing_dict,
            ),
            batch_size=self.hparams.batch_size,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def add_noise(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if (self.hparams.noise_factor == 0) or (not training):
            return x

        # expect input to be of shape [timesteps, bands]
        # and to be normalized with mean 0, std=1
        # if its not, it means no norm_dict was passed, so lets
        # just assume std=1
        noise = torch.normal(0, 1, size=x.shape).float() * self.hparams.noise_factor

        # the added noise is the same per band, so that the temporal relationships
        # are preserved
        # noise_per_timesteps = noise.repeat(x.shape[0], 1)
        return x + noise

    def _split_preds_and_get_loss(
        self, batch, add_preds: bool, loss_label: str, log_loss: bool, mode: str, batch_idx=int
    ) -> Dict:

        x = batch
        
        if mode == 'training':
            # Assumes the data input is 12 month long (time=12), so takes the last 12 months
            # of Arizona dataset
            assert x.shape in {(self.hparams.batch_size, 12, 12), (self.train_count % self.hparams.batch_size, 12, 12)}, x.shape

        input_to_encode = x[:, : self.hparams.input_months, :]

        # we will predict every timestep except the first one
        output_to_predict = x[:, 1:, :]
        encoder_output = self(input_to_encode)
        encoder_loss = self.forecaster_loss(encoder_output, output_to_predict)
        loss: Union[float, torch.Tensor] = encoder_loss

        if mode == 'training':
            print(f"[EPOCH {self.current_epoch}] Train Loss of Minibatch #{batch_idx}: {loss}")
            self.logger.experiment.add_scalar('Train loss', loss, self.log_recorder['train'])
            self.log_recorder['train'] += 1
        elif mode == 'validation':
            print(f"[EPOCH {self.current_epoch}] Validation Loss of Minibatch #{batch_idx}: {loss}")
            self.logger.experiment.add_scalar('Validation loss', loss, self.log_recorder['validation'])
            self.log_recorder['validation'] += 1
        elif mode == 'test':
            print(f"[EPOCH {self.current_epoch}] Test Loss of Minibatch #{batch_idx}: {loss}")
            self.logger.experiment.add_scalar('Test loss', loss, self.log_recorder['test'])
            self.log_recorder['test'] += 1

        output_dict = {
            loss_label: loss
        }
        
        if add_preds:
            output_dict.update(
                {
                    "encoder_prediction": encoder_output,
                    "encoder_target": output_to_predict,
                }
            )

        if log_loss:
            output_dict["log"] = {
                loss_label: loss
            }

        return output_dict

    def training_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=False, loss_label="loss", log_loss=True, mode='training', batch_idx=batch_idx
        )

    def validation_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="val_loss", log_loss=True, mode='validation', batch_idx=batch_idx
        )

    def test_step(self, batch, batch_idx):
        return self._split_preds_and_get_loss(
            batch, add_preds=True, loss_label="loss", log_loss=True, mode='test', batch_idx=batch_idx
        )

    def get_dataset(
        self,
        subset: str,
        normalizing_dict: Optional[Dict] = None,
        cache: Optional[bool] = None,
    ) -> ForecasterDataset:
        return ForecasterDataset(
            data_folder=Path(self.hparams.processed_data_folder),
            subset=subset,
            normalizing_dict=normalizing_dict,
            cache=self.hparams.cache if cache is None else cache
        )

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        self.logger.experiment.add_scalar("Train Loss (per epoch)", avg_loss, self.current_epoch)

        return {
            'loss': avg_loss
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        self.logger.experiment.add_scalar("Validation Loss (per epoch)", val_loss, self.current_epoch)

        return {
            'val_loss': val_loss
        }

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        self.logger.experiment.add_scalar("Test Loss (per epoch)", test_loss, self.current_epoch)

        return {
            'test_loss': test_loss
        }
        