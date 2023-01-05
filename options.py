import argparse

class Options():
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataroot", help="Path to dataset directory", default='data/food_c_data_sample')
        parser.add_argument("--ckpts_dir", help="Path to checkpoints directory", default='ckpts')
        parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
        parser.add_argument("--phase", help="Which mode (train/test) to run the program in?", default='train')
        parser.add_argument("--target_h", help="Height which each image will be resized to", default=224, type=int)
        parser.add_argument("--target_w", help="Width which each image will be resized to", default=224, type=int)
        parser.add_argument("--labels2idx_file", help="Path to file containing labels2idx mapping", default = 'data/labels2idx.json')
        parser.add_argument("--idx2labels_file", help="Path to file containing idx2labels mapping", default = 'data/idx2labels.json')
        parser.add_argument("--num_labels", help="total number of labels in the dataset (last layer size depends on this)", type=int, default=62)
        parser.add_argument("--lr", help="learning rate", type=float, default=0.001)
        parser.add_argument("--momentum", help="Momentum for gradient update", type=float, default=0.9)
        parser.add_argument("--num_epoch", help="Maximum number of epochs", type=int, default=100)
        parser.add_argument("--model_name", help="Which model backbone to use?", default='vanilla')
        return parser

    def gather_options(self):
        parser = self.initialize()
        return parser.parse_args()

    def print_options(self, args):
        for k,v in vars(args).items():
            print(f"{k}: {v}")

    def save_options(self, args, file_name):
        with open(file_name, "w") as f:
            for k,v in vars(args).items():
                f.write(f"{k}: {v}\n")
    
    def parse(self):
        args = self.gather_options()
        return args    
