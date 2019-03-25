import matplotlib.pyplot as plt

def PlotHistory(H):
    try:
        history_dict = H.history
        print history_dict.keys()
        acc = H.history['acc']
        N = range(1, len(acc) + 1)
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H.history["loss"], label="train_loss")
        plt.plot(N, H.history["acc"], label="train_acc")
        plt.title("Training Loss and Accuracy (Simple NN)")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig("data/train_chart_loss_acc.png")
        return 0
    except Exception, e:
        print 'Error gen plot!'
        print e
        return 0