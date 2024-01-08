python main.py --arch lstm --cvss_col access_vector --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col access_complexity --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col authentication --num_classes 2 --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col confidentiality --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col integrity --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col availability --early_stopping_metrics mcc
python main.py --arch lstm --cvss_col severity --early_stopping_metrics mcc

python main.py --arch gru --cvss_col access_vector --early_stopping_metrics mcc
python main.py --arch gru --cvss_col access_complexity --early_stopping_metrics mcc
python main.py --arch gru --cvss_col authentication --num_classes 2 --early_stopping_metrics mcc
python main.py --arch gru --cvss_col confidentiality --early_stopping_metrics mcc
python main.py --arch gru --cvss_col integrity --early_stopping_metrics mcc
python main.py --arch gru --cvss_col availability --early_stopping_metrics mcc
python main.py --arch gru --cvss_col severity --early_stopping_metrics mcc