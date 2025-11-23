import os
import argparse
import random
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None


def gather_images(images_dir):
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    imgs = [p for p in Path(images_dir).glob(
        '**/*') if p.suffix.lower() in exts]
    return imgs


def write_data_yaml(path, data_dir, names=['plate']):
    # Write YAML pointing at dataset directories (preferred by ultralytics)
    rel = os.path.relpath(data_dir)
    images_dir = os.path.join(rel, 'images')
    train_path = 'images/train' if os.path.isdir(
        os.path.join(images_dir, 'train')) else 'images'
    val_path = 'images/val' if os.path.isdir(
        os.path.join(images_dir, 'val')) else 'images'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"path: {rel}\n")
        f.write(f"train: {train_path}\n")
        f.write(f"val: {val_path}\n")
        f.write('names:\n')
        for n in names:
            f.write(f"  - {n}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 on local train_data (images+labels)')
    parser.add_argument('--data-dir', default='train_data',
                        help='Dataset folder containing `images` and `labels`')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fraction for validation split when no val set present')
    parser.add_argument('--model', default='yolov8n.pt',
                        help='Base model to start from')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='auto',
                        help="Device to use for training: 'auto', 'cpu', or GPU id like '0' or 'cuda:0'")
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint (preserve optimizer state when possible)')
    parser.add_argument('--copy', action='store_true',
                        help='Copy files when creating val split instead of moving')
    args = parser.parse_args()

    images_dir = os.path.join(args.data_dir, 'images')
    labels_dir = os.path.join(args.data_dir, 'labels')

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        print('Erwarte Ordner structure: <data-dir>/images und <data-dir>/labels')
        return

    images_train = os.path.join(images_dir, 'train')
    images_val = os.path.join(images_dir, 'val')
    labels_train = os.path.join(labels_dir, 'train')
    labels_val = os.path.join(labels_dir, 'val')

    # Case A: already have images/train + labels/train but no images/val -> create val split
    if os.path.isdir(images_train) and os.path.isdir(labels_train) and not os.path.isdir(images_val):
        print('Gefundene Struktur images/train + labels/train, erstelle Val-Split...')
        all_imgs = [p for p in Path(images_train).glob(
            '*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        if not all_imgs:
            print('Keine Bilder in', images_train)
            return
        random.shuffle(all_imgs)
        k = int(len(all_imgs) * (1 - args.val_split))
        train_imgs = all_imgs[:k]
        val_imgs = all_imgs[k:]

        os.makedirs(images_val, exist_ok=True)
        os.makedirs(labels_val, exist_ok=True)

        moved = 0
        skipped = 0
        for p in val_imgs:
            stem = p.stem
            src_img = p
            src_lbl = Path(labels_train) / f"{stem}.txt"
            if not src_lbl.exists():
                skipped += 1
                continue
            dest_img = Path(images_val) / p.name
            dest_lbl = Path(labels_val) / f"{stem}.txt"
            if args.copy:
                shutil.copy(src_img, dest_img)
                shutil.copy(src_lbl, dest_lbl)
            else:
                shutil.move(src_img, dest_img)
                shutil.move(src_lbl, dest_lbl)
            moved += 1
        print(
            f'Val-Split: verschoben/copied {moved} Paare, {skipped} ohne Label übersprungen')

    # Case B: images/ contains raw images (not split) -> split into images/train & images/val
    elif not os.path.isdir(images_train) and not os.path.isdir(images_val):
        print('Keine train/val Unterordner gefunden — erstelle train/val und splitte Bilder...')
        imgs = gather_images(images_dir)
        if not imgs:
            print('Keine Bilder im', images_dir)
            return
        # Only keep images that have matching label files in labels_dir
        paired = []
        for p in imgs:
            stem = Path(p).stem
            lbl = Path(labels_dir) / f"{stem}.txt"
            if lbl.exists():
                paired.append(p)
        if not paired:
            print('Keine Bild/Label Paare gefunden in', images_dir)
            return
        random.shuffle(paired)
        k = int(len(paired) * (1 - args.val_split))
        train_imgs = paired[:k]
        val_imgs = paired[k:]

        os.makedirs(images_train, exist_ok=True)
        os.makedirs(images_val, exist_ok=True)
        os.makedirs(labels_train, exist_ok=True)
        os.makedirs(labels_val, exist_ok=True)

        moved = 0
        for src in train_imgs:
            stem = Path(src).stem
            src_lbl = Path(labels_dir) / f"{stem}.txt"
            dest_img = Path(images_train) / Path(src).name
            dest_lbl = Path(labels_train) / f"{stem}.txt"
            if args.copy:
                shutil.copy(src, dest_img)
                shutil.copy(src_lbl, dest_lbl)
            else:
                shutil.move(src, dest_img)
                shutil.move(src_lbl, dest_lbl)
            moved += 1
        moved_val = 0
        for src in val_imgs:
            stem = Path(src).stem
            src_lbl = Path(labels_dir) / f"{stem}.txt"
            dest_img = Path(images_val) / Path(src).name
            dest_lbl = Path(labels_val) / f"{stem}.txt"
            if args.copy:
                shutil.copy(src, dest_img)
                shutil.copy(src_lbl, dest_lbl)
            else:
                shutil.move(src, dest_img)
                shutil.move(src_lbl, dest_lbl)
            moved_val += 1
        print(
            f'Erstellt train/val: {moved} train, {moved_val} val (kopiert={args.copy})')

    else:
        print('Benutze vorhandene Struktur (images/train & images/val erwartet).')

    data_yaml_path = os.path.join(args.data_dir, 'data.generated.yaml')
    write_data_yaml(data_yaml_path, args.data_dir, names=['plate'])
    print('Geschriebene data yaml:', data_yaml_path)

    if YOLO is None:
        print("Ultralytics nicht importierbar. Installiere `ultralytics` und `torch` und versuche es erneut.")
        return

    print('Starte Training mit Modell', args.model)
    try:
        model = YOLO(args.model)
        # report device info
        if torch is not None:
            try:
                cuda_avail = torch.cuda.is_available()
                print('torch.cuda.is_available():', cuda_avail)
                if cuda_avail:
                    print('CUDA version:', torch.version.cuda)
                    print('GPU count:', torch.cuda.device_count())
                    try:
                        print('GPU name:', torch.cuda.get_device_name(0))
                    except Exception:
                        pass
            except Exception:
                pass
        # pass resume flag to ultralytics (True preserves optimizer state when resuming a previous run)
        model.train(data=data_yaml_path, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, save=True, save_period=1, resume=args.resume, device=args.device)
        best = Path('runs')
        if best.exists():
            for p in best.rglob('best.pt'):
                dest = Path('best.pt')
                shutil.copy(p, dest)
                print('best.pt kopiert nach', dest)
                break
    except Exception as e:
        print('Fehler beim Training:', e)


if __name__ == '__main__':
    main()
