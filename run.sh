#python main.py --cvss_col access_vector
#python main.py --cvss_col access_complexity
#python main.py --cvss_col authentication --num_classes 2
#python main.py --cvss_col confidentiality
#python main.py --cvss_col integrity
#python main.py --cvss_col availability
#python main.py --cvss_col severity

#python main.py --arch cnn --cvss_col access_vector
#python main.py --arch cnn --cvss_col access_complexity
#python main.py --arch cnn --cvss_col authentication --num_classes 2
#python main.py --arch cnn --cvss_col confidentiality
#python main.py --arch cnn --cvss_col integrity
#python main.py --arch cnn --cvss_col availability
#python main.py --arch cnn --cvss_col severity

python main.py --arch lstm --cvss_col access_vector
