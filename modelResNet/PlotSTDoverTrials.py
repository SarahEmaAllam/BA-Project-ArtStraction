import numpy as np
import matplotlib.pyplot as plt


def find_longest_list(list):
    max = 0
    for l in list:
        if len(l) > max:
            max = len(l)

    print(max)
    return max


# list_trials_acc = np.load('list_trials_acc')
list_trials_val_acc = np.load('list_trials_val_acc.npy', allow_pickle=True)
# list_trials_loss = np.load('list_trials_loss')
list_trials_val_loss = np.load('list_trials_val_loss.npy', allow_pickle=True)
print(list_trials_val_acc)
print(list_trials_val_loss)
# print("initial", list_trials_val_acc)
num_elements_in_longest_list = find_longest_list(list_trials_val_acc)

res = list(zip(*list_trials_val_acc))
res_loss = list(zip(*list_trials_val_loss))
acc_standard_deviations_per_epoch = []
loss_standard_deviations_per_epoch = []
means_acc = []
means_loss= []

print("res list" , res)
for epoch in range(num_elements_in_longest_list-1):
  # print("res per epoch" , res[epoch])
  acc_std = np.std(res[epoch])
  print("acc std: " , acc_std)
  acc_standard_deviations_per_epoch.append(acc_std)

  print("res per epoch", res_loss[epoch])
  loss_std = np.std(res_loss[epoch])
  print("loss std: ", loss_std)
  loss_standard_deviations_per_epoch.append(loss_std)
  print("meansacc", np.mean(res[epoch]))
  mean_acc = np.mean(res[epoch])
  means_acc.append(mean_acc)
  print("means loss", np.mean(res_loss[epoch]))
  mean_loss = np.mean(res_loss[epoch])
  means_loss.append(mean_loss)


time = range(1, len(acc_standard_deviations_per_epoch)+1)
print("time", len(time))
print("list_trials_val_acc", len(means_acc))

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
# plt.plot(time,history.history['accuracy'], label='Training Accuracy')

plt.errorbar(time, means_acc, yerr=acc_standard_deviations_per_epoch, label='Validation Accuracy Over Trials')
# plt.errorbar(time, list_trials_val_loss, yerr=loss_standard_deviations_per_epoch, label='Validation Accuracy Over Trials')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.xlabel('epoch')
plt.title('Validation Accuracy')

plt.subplot(2, 1, 2)
# plt.plot(time, x_loss, label='Training Loss')
# plt.plot(time, list_val_loss, label='Validation Loss')
plt.errorbar(time, means_loss, yerr=loss_standard_deviations_per_epoch, label='Validation Loss Over Trials')
plt.legend(loc='upper right')
plt.ylim([0, 1.1])
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.xlabel('epoch')
plt.savefig('STDValAccLossOverTrials' )
plt.figure()
  # candidates = [list_value for list_value in list_trials_val_acc if index_in_list(list_value, epoch)]
  # print("list_trials_val_acc[:][epoch]", candidates[:][epoch])
  # print(candidates)
  # results_per_epoch.append(candidates[:][epoch])

# print("res", results_per_epoch)
# print("results_per_epoch[0]", results_per_epoch[0])
