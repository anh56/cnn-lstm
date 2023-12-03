# default runs
#python main.py --cvss_col access_vector
#python main.py --cvss_col access_complexity
#python main.py --cvss_col authentication --num_classes 2
#python main.py --cvss_col confidentiality
#python main.py --cvss_col integrity
#python main.py --cvss_col availability
#python main.py --cvss_col severity
#
python main.py --arch cnn --cvss_col access_vector
python main.py --arch cnn --cvss_col access_complexity
python main.py --arch cnn --cvss_col authentication --num_classes 2
python main.py --arch cnn --cvss_col confidentiality
python main.py --arch cnn --cvss_col integrity
python main.py --arch cnn --cvss_col availability
python main.py --arch cnn --cvss_col severity
#
python main.py --arch lstm --cvss_col access_vector
python main.py --arch lstm --cvss_col access_complexity
python main.py --arch lstm --cvss_col authentication --num_classes 2
python main.py --arch lstm --cvss_col confidentiality
python main.py --arch lstm --cvss_col integrity
python main.py --arch lstm --cvss_col availability
python main.py --arch lstm --cvss_col severity
## es runs f1
#python main.py --cvss_col access_vector --early_stopping_metrics f1
#python main.py --cvss_col access_complexity --early_stopping_metrics f1
#python main.py --cvss_col authentication --num_classes 2 --early_stopping_metrics f1
#python main.py --cvss_col confidentiality --early_stopping_metrics f1
#python main.py --cvss_col integrity --early_stopping_metrics f1
#python main.py --cvss_col availability --early_stopping_metrics f1
#python main.py --cvss_col severity --early_stopping_metrics f1
#
python main.py --arch cnn --cvss_col access_vector --early_stopping_metrics f1
python main.py --arch cnn --cvss_col access_complexity --early_stopping_metrics f1
python main.py --arch cnn --cvss_col authentication --num_classes 2 --early_stopping_metrics f1
python main.py --arch cnn --cvss_col confidentiality --early_stopping_metrics f1
python main.py --arch cnn --cvss_col integrity --early_stopping_metrics f1
python main.py --arch cnn --cvss_col availability --early_stopping_metrics f1
python main.py --arch cnn --cvss_col severity --early_stopping_metrics f1
#
python main.py --arch lstm --cvss_col access_vector --early_stopping_metrics f1
python main.py --arch lstm --cvss_col access_complexity --early_stopping_metrics f1
python main.py --arch lstm --cvss_col authentication --num_classes 2 --early_stopping_metrics f1
python main.py --arch lstm --cvss_col confidentiality --early_stopping_metrics f1
python main.py --arch lstm --cvss_col integrity --early_stopping_metrics f1
python main.py --arch lstm --cvss_col availability --early_stopping_metrics f1
python main.py --arch lstm --cvss_col severity --early_stopping_metrics f1
## es run mcc
#python main.py --cvss_col access_vector --early_stopping_metrics mcc
#python main.py --cvss_col access_complexity --early_stopping_metrics mcc
#python main.py --cvss_col authentication --num_classes 2 --early_stopping_metrics mcc
#python main.py --cvss_col confidentiality --early_stopping_metrics mcc
#python main.py --cvss_col integrity --early_stopping_metrics mcc
#python main.py --cvss_col availability --early_stopping_metrics mcc
#python main.py --cvss_col severity --early_stopping_metrics mcc
#
python main.py --arch cnn --cvss_col access_vector --early_stopping_metrics mcc
python main.py --arch cnn --cvss_col access_complexity --early_stopping_metrics mcc
python main.py --arch cnn --cvss_col authentication --num_classes 2 --early_stopping_metrics mcc
python main.py --arch cnn --cvss_col confidentiality --early_stopping_metrics mcc
python main.py --arch cnn --cvss_col integrity --early_stopping_metrics mcc
python main.py --arch cnn --cvss_col availability --early_stopping_metrics mcc
python main.py --arch cnn --cvss_col severity --early_stopping_metrics mcc
#
python main.py --arch lstm --cvss_col access_vector --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col access_complexity --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col authentication --num_classes 2 --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col confidentiality --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col integrity --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col availability --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col severity --early_stopping_metrics mcc
### runs with cross entropy
## default runs
#python main.py --loss_fn cross_entropy --cvss_col access_vector
#python main.py --loss_fn cross_entropy --cvss_col access_complexity
#python main.py --loss_fn cross_entropy --cvss_col authentication --num_classes 2
#python main.py --loss_fn cross_entropy --cvss_col confidentiality
#python main.py --loss_fn cross_entropy --cvss_col integrity
#python main.py --loss_fn cross_entropy --cvss_col availability
#python main.py --loss_fn cross_entropy --cvss_col severity
#
python main.py --loss_fn cross_entropy --arch cnn --cvss_col access_vector
python main.py --loss_fn cross_entropy --arch cnn --cvss_col access_complexity
python main.py --loss_fn cross_entropy --arch cnn --cvss_col authentication --num_classes 2
python main.py --loss_fn cross_entropy --arch cnn --cvss_col confidentiality
python main.py --loss_fn cross_entropy --arch cnn --cvss_col integrity
python main.py --loss_fn cross_entropy --arch cnn --cvss_col availability
python main.py --loss_fn cross_entropy --arch cnn --cvss_col severity
#
python main.py --loss_fn cross_entropy --arch lstm --cvss_col access_vector
python main.py --loss_fn cross_entropy --arch lstm --cvss_col access_complexity
python main.py --loss_fn cross_entropy --arch lstm --cvss_col authentication --num_classes 2
python main.py --loss_fn cross_entropy --arch lstm --cvss_col confidentiality
python main.py --loss_fn cross_entropy --arch lstm --cvss_col integrity
python main.py --loss_fn cross_entropy --arch lstm --cvss_col availability
python main.py --loss_fn cross_entropy --arch lstm --cvss_col severity
## es runs f1
#python main.py --loss_fn cross_entropy --cvss_col access_vector --early_stopping_metrics f1
#python main.py --loss_fn cross_entropy --cvss_col access_complexity --early_stopping_metrics f1
#python main.py --loss_fn cross_entropy --cvss_col authentication --num_classes 2 --early_stopping_metrics f1
#python main.py --loss_fn cross_entropy --cvss_col confidentiality --early_stopping_metrics f1
#python main.py --loss_fn cross_entropy --cvss_col integrity --early_stopping_metrics f1
#python main.py --loss_fn cross_entropy --cvss_col availability --early_stopping_metrics f1
#python main.py --loss_fn cross_entropy --cvss_col severity --early_stopping_metrics f1
#
python main.py --loss_fn cross_entropy --arch cnn --cvss_col access_vector --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch cnn --cvss_col access_complexity --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch cnn --cvss_col authentication --num_classes 2 --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch cnn --cvss_col confidentiality --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch cnn --cvss_col integrity --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch cnn --cvss_col availability --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch cnn --cvss_col severity --early_stopping_metrics f1
#
python main.py --loss_fn cross_entropy --arch lstm --cvss_col access_vector --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch lstm --cvss_col access_complexity --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch lstm --cvss_col authentication --num_classes 2 --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch lstm --cvss_col confidentiality --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch lstm --cvss_col integrity --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch lstm --cvss_col availability --early_stopping_metrics f1
python main.py --loss_fn cross_entropy --arch lstm --cvss_col severity --early_stopping_metrics f1
## es run mcc
#python main.py --loss_fn cross_entropy --cvss_col access_vector --early_stopping_metrics mcc
#python main.py --loss_fn cross_entropy --cvss_col access_complexity --early_stopping_metrics mcc
#python main.py --loss_fn cross_entropy --cvss_col authentication --num_classes 2 --early_stopping_metrics mcc
#python main.py --loss_fn cross_entropy --cvss_col confidentiality --early_stopping_metrics mcc
#python main.py --loss_fn cross_entropy --cvss_col integrity --early_stopping_metrics mcc
#python main.py --loss_fn cross_entropy --cvss_col availability --early_stopping_metrics mcc
#python main.py --loss_fn cross_entropy --cvss_col severity --early_stopping_metrics mcc
#
python main.py --loss_fn cross_entropy --arch cnn --cvss_col access_vector --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch cnn --cvss_col access_complexity --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch cnn --cvss_col authentication --num_classes 2 --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch cnn --cvss_col confidentiality --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch cnn --cvss_col integrity --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch cnn --cvss_col availability --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch cnn --cvss_col severity --early_stopping_metrics mcc
#
python main.py --loss_fn cross_entropy --arch lstm --cvss_col access_vector --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch lstm --cvss_col access_complexity --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch lstm --cvss_col authentication --num_classes 2 --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch lstm --cvss_col confidentiality --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch lstm --cvss_col integrity --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch lstm --cvss_col availability --early_stopping_metrics mcc
python main.py --loss_fn cross_entropy --arch lstm --cvss_col severity --early_stopping_metrics mcc
