import numpy as np
import h5py


def creat_dataset():
    datas = {}
    for i in range(8):
        filename = 'data/data60to30/dataset{}.npz'.format(i + 1)
        dataset = np.load(filename)
        datas["pos{}".format(i)] = dataset['pos'].astype(np.float32)
        datas["neg{}".format(i)] = dataset['neg'].astype(np.float32)

    postrain = datas["pos0"]
    negtrain = datas["neg0"]
    for i in range(1, 6):
        postrain = np.concatenate((postrain, datas["pos{}".format(i)]), axis=0)
        negtrain = np.concatenate((negtrain, datas["neg{}".format(i)]), axis=0)
    postest = np.concatenate((datas["pos6"], datas["pos7"]), axis=0)
    negtest = np.concatenate((datas["neg6"], datas["neg7"]), axis=0)

    xtrain_orig = np.concatenate((postrain, negtrain), axis=0)
    xtest_orig = np.concatenate((postest, negtest), axis=0)
    ytrain = np.append(np.ones(np.shape(postrain)[0], dtype=int),
                       np.zeros(np.shape(negtrain)[0], dtype=int))
    ytest = np.append(np.ones(np.shape(postest)[0], dtype=int),
                      np.zeros(np.shape(negtest)[0], dtype=int))

    randTrainPerm = np.random.choice(np.arange(len(ytrain)), size=len(ytrain), replace=False)
    xtrain_orig = xtrain_orig[randTrainPerm]
    ytrain = ytrain[randTrainPerm]
    classes = [0,1]
    file = h5py.File('data/data60to30/data60to30.h5', 'w')
    file.create_dataset('xtrain_orig', data=xtrain_orig)
    file.create_dataset('ytrain', data=ytrain)
    file.create_dataset('xtest_orig', data=xtest_orig)
    file.create_dataset('ytest', data=ytest)
    file.create_dataset('classes', data=classes)
    file.close()
    # return xtrain_orig,ytrain,xtest_orig,ytest,classes


def load_dataset(filename='data/data60to30/data60to30.h5'):
    file = h5py.File(filename, 'r')
    xtrain_orig = np.array(file['xtrain_orig'][:])
    ytrain = np.array(file['ytrain'][:])
    ytrain = ytrain.reshape(1,len(ytrain))
    xtest_orig = np.array(file['xtest_orig'][:])
    ytest = np.array(file['ytest'][:])
    ytest = ytest.reshape(1,len(ytest))
    classes = np.array(file['classes'][:])
    return xtrain_orig, ytrain, xtest_orig, ytest, classes


if __name__ == '__main__':
    #creat_dataset()
    xtrain_orig, ytrain, xtest_orig, ytest, classes = load_dataset()
    print("test down")

