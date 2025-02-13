import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers.optimization import AdamW
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoFeatureExtractor, HubertForSequenceClassification, AutoConfig

# 환경 변수 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Mixed Precision 설정
precision = '16-mixed'
def accuracy(preds, labels):
    return (preds == labels).float().mean()

def getAudios(df):
    audios = []
    labels = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        try:
            audio, _ = librosa.load(row['path'], sr=SAMPLING_RATE)
            audios.append(audio)
            label = row['label']
            if 0 <= label < NUM_LABELS:
                labels.append(label)
            else:
                print(f"Invalid label {label} at row {idx}, skipping.")
        except FileNotFoundError:
            print(f"File not found: {row['path']}, skipping.")
    return audios, labels

class MyDataset(Dataset):
    def __init__(self, audio, audio_feature_extractor, label=None):
        if label is None:
            label = [0] * len(audio)
        self.label = np.array(label).astype(np.int64)
        self.audio = audio
        self.audio_feature_extractor = audio_feature_extractor

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        if not (0 <= label < NUM_LABELS):
            raise ValueError(f"Invalid label {label} at index {idx}")
        audio = self.audio[idx]
        audio_feature = self.audio_feature_extractor(raw_speech=audio, return_tensors='np', sampling_rate=SAMPLING_RATE)
        audio_values, audio_attn_mask = audio_feature['input_values'][0], audio_feature['attention_mask'][0]

        item = {
            'label': label,
            'audio_values': audio_values,
            'audio_attn_mask': audio_attn_mask,
        }

        return item


def collate_fn(samples):
    batch_labels = []
    batch_audio_values = []
    batch_audio_attn_masks = []

    for sample in samples:
        batch_labels.append(sample['label'])
        batch_audio_values.append(torch.tensor(sample['audio_values']))
        batch_audio_attn_masks.append(torch.tensor(sample['audio_attn_mask']))

    batch_labels = torch.tensor(batch_labels)
    batch_audio_values = pad_sequence(batch_audio_values, batch_first=True)
    batch_audio_attn_masks = pad_sequence(batch_audio_attn_masks, batch_first=True)

    batch = {
        'label': batch_labels,
        'audio_values': batch_audio_values,
        'audio_attn_mask': batch_audio_attn_masks,
    }

    return batch

class MyLitModel(pl.LightningModule):
    def __init__(self, audio_model_name, num_labels, n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=1):
        super(MyLitModel, self).__init__()
        self.config = AutoConfig.from_pretrained(audio_model_name)
        self.config.num_labels = num_labels
        self.config.activation_dropout = dropout
        self.config.attention_dropout = dropout
        self.config.final_dropout = dropout
        self.config.hidden_dropout = dropout
        self.config.hidden_dropout_prob = dropout
        self.audio_model = HubertForSequenceClassification.from_pretrained(audio_model_name, config=self.config)
        self.lr_decay = lr_decay
        self._do_reinit(n_layers, projector, classifier)

    def forward(self, audio_values, audio_attn_mask):
        logits = self.audio_model(input_values=audio_values, attention_mask=audio_attn_mask).logits
        if logits.size(-1) != NUM_LABELS:
            raise ValueError(f"Output logits dimension {logits.size(-1)} does not match NUM_LABELS {NUM_LABELS}")
        return logits

    def training_step(self, batch, batch_idx):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        labels = batch['label']

        logits = self(audio_values, audio_attn_mask)

        loss = nn.CrossEntropyLoss()(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        labels = batch['label']

        logits = self(audio_values, audio_attn_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']

        logits = self(audio_values, audio_attn_mask)
        preds = torch.argmax(logits, dim=1)

        return preds

    def configure_optimizers(self):
        lr = 1e-5
        layer_decay = self.lr_decay
        weight_decay = 0.01
        llrd_params = self._get_llrd_params(lr=lr, layer_decay=layer_decay, weight_decay=weight_decay)
        optimizer = AdamW(llrd_params)
        return optimizer

    def _get_llrd_params(self, lr, layer_decay, weight_decay):
        n_layers = self.audio_model.config.num_hidden_layers
        llrd_params = []
        for name, value in list(self.named_parameters()):
            if ('bias' in name) or ('layer_norm' in name):
                llrd_params.append({"params": value, "lr": lr, "weight_decay": 0.0})
            elif ('emb' in name) or ('feature' in name):
                llrd_params.append(
                    {"params": value, "lr": lr * (layer_decay ** (n_layers + 1)), "weight_decay": weight_decay})
            elif 'encoder.layer' in name:
                for n_layer in range(n_layers):
                    if f'encoder.layer.{n_layer}' in name:
                        llrd_params.append({"params": value, "lr": lr * (layer_decay ** (n_layer + 1)),
                                            "weight_decay": weight_decay})
            else:
                llrd_params.append({"params": value, "lr": lr, "weight_decay": weight_decay})
        return llrd_params

    def _do_reinit(self, n_layers=0, projector=True, classifier=True):
        if projector:
            self.audio_model.projector.apply(self._init_weight_and_bias)
        if classifier:
            self.audio_model.classifier.apply(self._init_weight_and_bias)

        for n in range(n_layers):
            self.audio_model.hubert.encoder.layers[-(n + 1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.audio_model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

DATA_DIR = 'finedata'
MODEL_DIR = 'finemodel'
SAMPLING_RATE = 16000
SEED = 0
N_FOLD = 5
BATCH_SIZE = 2 #8
NUM_LABELS = 7  # 클래스 범위 0-6

def train_model():
    seed_everything(SEED)

    audio_model_name = 'Rajaram1996/Hubert_emotion'
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
    audio_feature_extractor.return_attention_mask = True

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    train_df['path'] = train_df['path'].apply(lambda x: os.path.join(DATA_DIR, 'train', os.path.basename(x)))
    test_df['path'] = test_df['path'].apply(lambda x: os.path.join(DATA_DIR, 'test', os.path.basename(x)))
    train_audios, train_label = getAudios(train_df)
    test_audios, test_label = getAudios(test_df)

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(train_label, train_label)):
        train_fold_audios = [train_audios[train_index] for train_index in train_indices]
        val_fold_audios = [train_audios[val_index] for val_index in val_indices]

        train_fold_label = train_label[train_indices]
        val_fold_label = train_label[val_indices]
        train_fold_ds = MyDataset(train_fold_audios, audio_feature_extractor, train_fold_label)
        val_fold_ds = MyDataset(val_fold_audios, audio_feature_extractor, val_fold_label)
        train_fold_dl = DataLoader(train_fold_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0)
        val_fold_dl = DataLoader(val_fold_ds, batch_size=BATCH_SIZE * 2, collate_fn=collate_fn, num_workers=0)

        checkpoint_acc_callback = ModelCheckpoint(
            monitor='val_acc',
            dirpath=MODEL_DIR,
            filename=f'fold_{fold_idx}_' + '_{epoch:02d}-{val_acc:.4f}-{train_acc:.4f}',
            save_top_k=1,
            mode='max'
        )

        my_lit_model = MyLitModel(
            audio_model_name=audio_model_name,
            num_labels=NUM_LABELS,
            n_layers=1, projector=True, classifier=True, dropout=0.07, lr_decay=0.8
        )

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=30,
            precision='16-mixed',
            val_check_interval=0.1,
            callbacks=[checkpoint_acc_callback],
        )

        trainer.fit(my_lit_model, train_fold_dl, val_fold_dl)

        del my_lit_model

    # Test predictions and accuracy calculation
    test_ds = MyDataset(test_audios, audio_feature_extractor, test_label)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, collate_fn=collate_fn)
    pretrained_models = list(map(lambda x: os.path.join(MODEL_DIR, x), os.listdir(MODEL_DIR)))

    test_preds = []
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        precision=16,
    )
    for pretrained_model_path in pretrained_models:
        pretrained_model = MyLitModel.load_from_checkpoint(
            pretrained_model_path,
            audio_model_name=audio_model_name,
            num_labels=NUM_LABELS,
        )
        test_pred = trainer.predict(pretrained_model, test_dl)
        test_pred = torch.cat(test_pred).detach().cpu().numpy()
        test_preds.append(test_pred)
        del pretrained_model

    # Calculate accuracy
    final_test_preds = np.argmax(np.mean(np.array(test_preds), axis=0), axis=1)
    test_accuracy = np.mean(final_test_preds == test_label)
    print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

def check_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

if __name__ == '__main__':
    check_device()
    train_model()
