import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # file = open('../1room_exp/1roomb_bn_lstm_fc_data.csv', 'r')
    file = open('../2room_exp/2room_bn_lstm_fcjun25_data.csv', 'r')
    # file = open('../1room_exp/1room_bn_fc_data.csv', 'r')
    # file = open('../2room_exp/2room_baselstm_data.csv', 'r')

    df = list(csv.reader(file, delimiter=','))
    # df = list(filter(lambda x: x, df))
    x = [int(d[0]) for d in df]
    y = [float(d[1].lstrip('tensor(')) for d in df]

    print(x)
    print(y)

    plt.plot(x, y)
    plt.xlabel('epochs')
    plt.ylabel('inference accuracy')
    plt.title('AlexNet + BN + LSTM + FC in 2 room')
    # plt.title('AACBase + BN + LSTM in 2 rooms')
    # plt.show()

    plt.savefig('../plots/alexbnlstmfc_2room.png')

    file.close()
    


