import os
import sys
import pickle
import inspect
import torch
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import build_vocab
from data_loader import get_loader
from archs import ContextAgnosticModel, ContextAwareModel, AttentionalContextAwareModel


def main(args, device, fields):
    threshold_UNK = 5  # words with frequency of less than this threshold will be UNK
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # Build and load vocabulary wrapper
    print('loading vocab...')
    if not os.path.exists(args.vocab_root):
        os.mkdir(args.vocab_root)
        label_vocab, context_vocab, category_vocab = build_vocab(args.train_content_path, threshold_UNK, args.glove_dim, fields)
        with open(os.path.join(args.vocab_root, "label.pkl"), 'wb') as f:
            pickle.dump(label_vocab, f)
        with open(os.path.join(args.vocab_root, "context.pkl"), 'wb') as f:
            pickle.dump(context_vocab, f)
        with open(os.path.join(args.vocab_root, "category.pkl"), 'wb') as f:
            pickle.dump(category_vocab, f)
    else:
        print(os.path.join(args.vocab_root, "label.pkl"))
        with open(os.path.join(args.vocab_root, "label.pkl"), 'rb') as f:
            label_vocab = pickle.load(f)
        with open(os.path.join(args.vocab_root, "context.pkl"), 'rb') as f:
            context_vocab = pickle.load(f)
        with open(os.path.join(args.vocab_root, "category.pkl"), 'rb') as f:
            category_vocab = pickle.load(f)

    print('vocab loaded! con_len:{}, context_len:{}'.format(len(label_vocab), len(context_vocab)))

    #  imagenet’s parameter mean and std
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(min(244, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Build data loader
    test_loader = get_loader(args.image_path, args.test_content_path, label_vocab, context_vocab, fields,
                             category_vocab,
                             transform=transform, batch_size=args.batch_size,
                             shuffle=True, num_workers=2)
    if args.train:
        train_loader = get_loader(args.image_path, args.train_content_path, label_vocab, context_vocab, fields, category_vocab,
                                  transform=transform, batch_size=args.batch_size,
                                  shuffle=True, num_workers=2, w_loss=args.weighted_loss, mask_threshold=0)

        val_loader = get_loader(args.image_path, args.val_content_path, label_vocab, context_vocab, fields, category_vocab,
                              transform=transform, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    model = None
    if not args.context_aware:
        model = ContextAgnosticModel(args.embed_size, label_vocab, device, args.hidden_size)
    else:
        if not args.attention:
            model = ContextAwareModel(args.embed_size, label_vocab, device, args.hidden_size,
                                      context_vocab, args.glove_dim, max(args.glove_dim, len(category_vocab)))
        else:
            model = AttentionalContextAwareModel(args.embed_size, label_vocab, device, args.hidden_size,
                                                 context_vocab, args.glove_dim, max(args.glove_dim, len(category_vocab)))
    if args.train:
        model.train(train_loader, val_loader, args.epochs, args.log_step, args.save_step, args.model_path, args.validation)
    else:
        print("loading model")
        model.load_model(args.model_path)
    print("Evaluating the model...")
    model.report_metrics(test_loader, args.image_path, args.log)


if __name__ == '__main__':
    import argparse

    DEFAULT_VOCAB_VERSION = 1
    DEFAULT_IMG_VERSION = 1
    DEFAULT_SPLIT = 1
    model_conf = 0
    _device = torch.device('cuda:{}'.format(model_conf) if torch.cuda.is_available() else 'cpu')
    print("Device: ", _device)
    parser = argparse.ArgumentParser(description='CoALa: content description generation')
    parser.add_argument('--vocab-root', type=str,
                        default="out/vocab_v{}".format(DEFAULT_VOCAB_VERSION), help='Vocab Root')
    parser.add_argument('--model-path', type=str,
                        default="out/models/v_{}.t_{}.{}".format(DEFAULT_VOCAB_VERSION, DEFAULT_SPLIT, model_conf),
                        help='Model Path')
    parser.add_argument('--image-path', type=str, required=True, help='Image Directory')
    parser.add_argument('--val-content-path', type=str, help='Validation Content Description Directory')
    parser.add_argument('--train-content-path', type=str, help='Training Content Description Directory')
    parser.add_argument('--test-content-path', type=str, required=True, help='Test Content Description Directory')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size')
    parser.add_argument('--embed-size', type=int, default=100, help='Embed Size')
    parser.add_argument('--hidden-size', type=int, default=100, help='Hidden Size')
    parser.add_argument('--image-size', type=int, default=256, help='Image Size')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--log-step', type=int, default=10, help='Log Step')
    parser.add_argument('--save-step', type=int, default=10, help='Save Step')
    parser.add_argument('--glove-dim', type=int, default=100, help='GloVe dimension')
    parser.add_argument('--no-context-aware', dest='context_aware', action='store_false', help='No Context-aware Mode')
    parser.add_argument('--no-attention', dest='attention', action='store_false', help='No Attention Mechanism Mode')
    parser.add_argument('--no-validation', dest='validation', action='store_false', help='No Validation')
    parser.add_argument('--weighted_loss', dest='weighted_loss', action='store_true', help='Weighted loss')
    parser.add_argument('--test', dest='train', action='store_false', help='Test Mode')
    parser.add_argument('--log', dest='log', action='store_true', help='copy image and results')
    parser.set_defaults(log=False)
    parser.set_defaults(train=True)
    parser.set_defaults(attention=True)
    parser.set_defaults(context_aware=True)
    parser.set_defaults(weighted_loss=False)
    _args = parser.parse_args()
    print("Arguments", _args)
    context_fields = ['title', 'activity_name', 'android_id', "father-id", "siblings-id", "siblings-text","location", "category"]
    main(_args, _device, context_fields)

