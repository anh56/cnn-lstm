# default multitask runs
python main.py --arch cnn --multitask
python main.py --arch lstm --multitask
python main.py --arch cnn --multitask --early_stopping_metrics f1
python main.py --arch lstm --multitask --early_stopping_metrics f1
python main.py --arch cnn --multitask --early_stopping_metrics mcc
python main.py --arch lstm --multitask --early_stopping_metrics mcc
python main.py --arch cnn --multitask --loss_fn cross_entropy
python main.py --arch lstm --multitask --loss_fn cross_entropy
python main.py --arch cnn --multitask --loss_fn cross_entropy --early_stopping_metrics f1
python main.py --arch lstm --multitask --loss_fn cross_entropy --early_stopping_metrics f1
python main.py --arch cnn --multitask --loss_fn cross_entropy --early_stopping_metrics mcc
python main.py --arch lstm --multitask --loss_fn cross_entropy --early_stopping_metrics mcc

python main.py --arch cnn --multitask --learning_rate 0.0001
python main.py --arch lstm --multitask --learning_rate 0.0001
python main.py --arch cnn --multitask --early_stopping_metrics f1 --learning_rate 0.0001
python main.py --arch lstm --multitask --early_stopping_metrics f1 --learning_rate 0.0001
python main.py --arch cnn --multitask --early_stopping_metrics mcc --learning_rate 0.0001
python main.py --arch lstm --multitask --early_stopping_metrics mcc --learning_rate 0.0001
python main.py --arch cnn --multitask --loss_fn cross_entropy --learning_rate 0.0001
python main.py --arch lstm --multitask --loss_fn cross_entropy --learning_rate 0.0001
python main.py --arch cnn --multitask --loss_fn cross_entropy --early_stopping_metrics f1 --learning_rate 0.0001
python main.py --arch lstm --multitask --loss_fn cross_entropy --early_stopping_metrics f1 --learning_rate 0.0001
python main.py --arch cnn --multitask --loss_fn cross_entropy --early_stopping_metrics mcc --learning_rate 0.0001
python main.py --arch lstm --multitask --loss_fn cross_entropy --early_stopping_metrics mcc --learning_rate 0.0001

python main.py --arch cnn --multitask --learning_rate 0.0005
python main.py --arch lstm --multitask --learning_rate 0.0005
python main.py --arch cnn --multitask --early_stopping_metrics f1 --learning_rate 0.0005
python main.py --arch lstm --multitask --early_stopping_metrics f1 --learning_rate 0.0005
python main.py --arch cnn --multitask --early_stopping_metrics mcc --learning_rate 0.0005
python main.py --arch lstm --multitask --early_stopping_metrics mcc --learning_rate 0.0005
python main.py --arch cnn --multitask --loss_fn cross_entropy --learning_rate 0.0005
python main.py --arch lstm --multitask --loss_fn cross_entropy --learning_rate 0.0005
python main.py --arch cnn --multitask --loss_fn cross_entropy --early_stopping_metrics f1 --learning_rate 0.0005
python main.py --arch lstm --multitask --loss_fn cross_entropy --early_stopping_metrics f1 --learning_rate 0.0005
python main.py --arch cnn --multitask --loss_fn cross_entropy --early_stopping_metrics mcc --learning_rate 0.0005
python main.py --arch lstm --multitask --loss_fn cross_entropy --early_stopping_metrics mcc --learning_rate 0.0005