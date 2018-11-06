import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file = open('bn_fc.csv', 'r')

    df = list(csv.reader(file, delimiter=','))
    df = list(filter(lambda x: x, df))
    x = [int(d[0])-5 for d in df]
    y = [float(d[1].lstrip('tensor(')) for d in df]

    print(x)
    print(y)

    plt.plot(x, y)
    plt.xlabel('epochs')
    plt.ylabel('inference accuracy')
    plt.title('AlexNet + BN + FC')
    plt.savefig('alexbnfc.png')

    file.close()
    


