import torch 
from pytorch_lightning import Trainer
from models.model import classifier
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
import pytorch_lightning as pl
from utils.dataloader import dataloader, DM, IEMOCAP, MER2023, MER4
import hydra
import wandb
import os 
import pandas as pd
import numpy as np
import time

@hydra.main(config_path="../conf", config_name="config.yaml",version_base=None)
def train(cfg):
    """ Training the VAE model using pytorch lightning. """
    # set up 
    hparams = cfg.experiments
    dparams = cfg.data
    cparams = cfg.collect
    mparams = cfg.model_name
    drparams = cfg.drop
    sparams = cfg.sam
    name = cparams.dataset+"_"+mparams.model_name+"_"+mparams.type+"_"+hparams.name+"_"+drparams.name+"_"+dparams.name#+"_"+sparams.name
    torch.manual_seed(hparams.seed)
    save_path = f"/work3/s194644/models/few/{name}_{sparams.samples}"
    multi = False
    if cparams.dataset =="iemocap":
        path = "/work3/s194644/iemocap/"
        dataset = IEMOCAP(path, batch_size=hparams.batch_size,num_class=cparams.num_classes,type1=mparams.type)
    elif cparams.dataset =="mer2024":
        path = "/work3/s194644/data/mer/"
        dataset = MER2023(path, batch_size=hparams.batch_size,num_class=cparams.num_classes,type1=mparams.type)
    elif cparams.dataset =="mer4":
        path = "/work3/s194644/data/mer/"
        dataset = MER4(path, batch_size=hparams.batch_size,num_class=cparams.num_classes,type1=mparams.type,no_samples=sparams.samples)
    train_loaders, eval_loaders, test_loaders = dataset.get_loaders()
    test_mse_scores_0, test_f1_scores_0 = [], []
    test_mse_scores_1, test_f1_scores_1 = [], []
    test_f1_scores_2 = []
    path = "/work3/s194644/models/"
    ex = "Zero-shot"
    name = name.replace('mer4', 'iemocap')
    weightpath = "/work3/s194644/models/Zero-shot/iemocap_cm1d_1d_exp5_drop4_data1"
    weights = os.listdir(weightpath)
    for ii in range(len(train_loaders)):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        eval_loader  = eval_loaders[ii]
        if cparams.dataset =="iemocap":
            test_loader = test_loaders[ii]
        else:
            test_loader = test_loaders
        for weight in weights:
            model = classifier.load_from_checkpoint(f"{weightpath}/{weight}",model_name=mparams.model_name,multi=multi,learning_rate=hparams.lr,h_dim=hparams.h_dim,num_classes=cparams.num_classes,dropout_prob=drparams.dropout,scheduler="test",scheduler_patience="test", scheduler_factor="test",a=dparams.a,t=dparams.t,v=dparams.v,type=mparams.type)
        
            # save best model 
            checkpoint_callback = ModelCheckpoint(dirpath=f"{save_path}", monitor="val_loss", mode="min")

            # add early stopping
            early_stopping_callback = EarlyStopping(monitor="val_loss", patience=50, verbose=True, mode="min")

            # monitor the learning rate 
            lr_monitor = LearningRateMonitor(logging_interval='epoch')

            # train
            torch.set_float32_matmul_precision('highest')
            trainer = Trainer(callbacks=[early_stopping_callback, checkpoint_callback,lr_monitor],#,StochasticWeightAveraging(swa_lrs=1e-4)],
                                logger=pl.loggers.WandbLogger(project="Master_thesis",name=name),
                                log_every_n_steps=10,
                                accelerator="gpu",
                                devices=1,
                                #gradient_clip_val=1.0,
                                max_epochs=hparams.max_epochs,
                                precision='16-mixed')
                                #profiler="advanced")

            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
            #test the best model
            model = classifier.load_from_checkpoint(checkpoint_callback.best_model_path,model_name=mparams.model_name,multi=multi,learning_rate=hparams.lr,h_dim=hparams.h_dim,num_classes=cparams.num_classes,dropout_prob=drparams.dropout,scheduler="test",scheduler_patience="test", scheduler_factor="test",a=dparams.a,t=dparams.t,v=dparams.v,type=mparams.type)
            test_results = trainer.test(model,test_loader)
            wandb.finish()

            if cparams.dataset =="iemocap":
                test_f1_scores_0.append(test_results[0]['test_f1'])
            else:
                test_f1_scores_0.append(test_results[0]['test_f1/dataloader_idx_0'])
                test_f1_scores_1.append(test_results[1]['test_f1/dataloader_idx_1'])
                test_f1_scores_2.append(test_results[2]['test_f1/dataloader_idx_2'])
    
    def mean_std_str(scores):
        return f"{np.mean(scores):.4f} ± {np.std(scores):.4f}"

    if cparams.dataset =="iemocap":
        results_df = pd.DataFrame({"experiment":name,
        "IEMOCAP f1":[mean_std_str(test_f1_scores_0)]})
    else:
        # Save metrics to CSV in the desired format
        results_df = pd.DataFrame({
            "experiment":name,
            "MER-MULTI f1": [mean_std_str(test_f1_scores_0)],
            "MER-NOISE f1": [mean_std_str(test_f1_scores_1)],
            "MER-SEMI f1": [mean_std_str(test_f1_scores_2)]
        })
    results_df.to_csv(f'/work3/s194644/results/few/{name}_{sparams.samples}.csv', index=False)
    print("Test metrics saved to test_metrics.csv")
if __name__ == "__main__":
    train() 