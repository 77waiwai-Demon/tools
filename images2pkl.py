import os
import random
import cv2
import pickle

from tqdm import tqdm


class BaseTrainData(object):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    ITEM_JSON = "item.json"
    def __init__(self, folder, resize_size):
        self.folder = folder
        self.resize_size = resize_size
        self.train_folder = os.path.join(folder, BaseTrainData.TRAIN) #/media/ubuntu/lihui/DING/fresh_crop_h_w/
        self.val_folder = os.path.join(folder, BaseTrainData.VAL)
        self.test_folder = os.path.join(folder, BaseTrainData.TEST)
        self.__check()
        self.ori_category = []

        # not_list = ['010020001', '010040001', '010060001', '010060003', '010090001', '010090005', '010090007', '010120002', '010130002', '010160001', '020060001', '020090001', '020090002', '020130014', '020210005', '020210007', '020210014', '020320001', '050040004', '050040009', '060020007', '080020033']
        not_list = []
        for cls_item in os.listdir(self.train_folder):
            if cls_item not in not_list:
                self.ori_category.append(cls_item)
        self.ori_category.sort()
        self.id_class_map = { idx:cls for idx, cls in enumerate(self.ori_category)}
        self.class_id_map = { cls:idx for idx, cls in enumerate(self.ori_category)}


    def __check(self):
        train_class = os.listdir(self.train_folder)
        val_class = os.listdir(self.val_folder)
        if len(set(train_class).difference(set(val_class))) != 0:
            raise Exception("train and val is not equal")

    # def dump_map_json(self, dst):


    @classmethod
    def save_train(self, dst):
        pass

    @classmethod
    def save_val(self, dst):
        pass

    @classmethod
    def save_test(self, dst):
        pass

class PickleTrainData(BaseTrainData):
    def __save(self, dst, flag, shuffle=True):
        data_dict = {'labels': [], 'data': []}
        folder = self.train_folder if flag == BaseTrainData.TRAIN else self.val_folder
        image_list = []
        for cls in self.ori_category:
            count = 0
            sub_folder = os.path.join(folder, cls)

            image_names = []
            pth =os.listdir(sub_folder)
            for name in os.listdir(sub_folder):
                count += 1
                image_names.append(os.path.join(sub_folder, name))
                if count == 10:
                    break
            # image_names = [os.path.join(sub_folder, name) for name in os.listdir(sub_folder)]
            image_list.extend(image_names)
        # print(image_list)
        if shuffle:
            random.shuffle(image_list)
        for image_path in tqdm(image_list):
            # print(image_path)
            image = cv2.imread(image_path)
            label = self.class_id_map[os.path.basename(os.path.dirname(image_path))]
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.resize_size)
            data_dict['labels'].append(label)
            data_dict['data'].append(img)
        save_folder = os.path.dirname(os.path.abspath(dst))
        os.makedirs(save_folder, exist_ok=True)
        with open(dst, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_train(self, dst):
        self.__save(dst, BaseTrainData.TRAIN, True)

    def save_val(self, dst):
        self.__save(dst, BaseTrainData.VAL, True)

    def save_test(self, dst):
        self.__save(dst, BaseTrainData.TEST, True)


if __name__ == "__main__":
    pickle_train_data = PickleTrainData(r'/media/ubuntu/lihui/DING/fresh_crop_h_w/reset_all', (256, 256))
    pickle_train_data.save_train(r'/media/ubuntu/lihui/DING/fresh_crop_h_w/reset_all/10_standard_images.pickle')
    #
    # pickle_train_data.save_val(r'/media/ubuntu/lihui/DING/fresh_crop_h_w/reset_all/treGoods_category_split_train_phase_test.pickle')
    # pickle_train_data.save_test(r'/media/ubuntu/lihui/DING/fresh_crop_h_w/reset_all/treGoods_category_split_train_phase_val.pickle')
#     print(pickle_train_data.id_class_map)
#     for i in range(217):
#         print(pickle_train_data.id_class_map[i])
