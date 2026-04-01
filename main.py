import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint

def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def load_dataset(dataset):
	folder = os.path.join(output_folder, dataset)
	if not os.path.exists(folder):
		raise Exception('Processed Data not found.')
	loader = []
	for file in ['train', 'test', 'labels']:
		if dataset == 'SMD': file = 'machine-1-1_' + file
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))
	if args.less: loader[0] = cut_array(0.2, loader[0])
	train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
	test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
	labels = loader[2]
	# print(f"train_loader: {train_loader}")
	# print(f"test_loader: {test_loader}")
	# print(f"labels: {labels}")
	return train_loader, test_loader, labels

def save_model(model, optimizer, scheduler, epoch, accuracy_list):
	folder = f'checkpoints/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/model.ckpt'
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
	
def load_model(modelname, dims):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
	fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
	if os.path.exists(fname) and (not args.retrain or args.test):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		# PyTorch 2.6+: default weights_only=True breaks checkpoints that store
		# non-tensor objects (e.g. numpy scalars in accuracy_list). Local ckpts only.
		try:
			checkpoint = torch.load(fname, weights_only=False)
		except TypeError:  # PyTorch < 2.0 without weights_only kwarg
			checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, scheduler, epoch, accuracy_list

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
	l = nn.MSELoss(reduction = 'none')
	feats = dataO.shape[1]
	# Ensure double dtype without reconstructing an already-tensor input incorrectly
	data_x = torch.DoubleTensor(data)
	# data_x = torch.as_tensor(data, dtype=torch.double)
	dataset = TensorDataset(data_x, data_x)
	bs = model.batch if training else len(data)
	dataloader = DataLoader(dataset, batch_size = bs, shuffle= None, num_workers= 2)
	# Paper hyperparameter (Sec 3.2): use a fixed η value.
	eta = 0.5	# 동일 가중 (임시)
	w_size = model.n_window
	l1s, l2s = [], []
	if training:
		def _set_requires_grad(mod, requires_grad: bool):
			if mod is None:
				return
			for p in mod.parameters():
				p.requires_grad = requires_grad

		for d, _ in dataloader:
			local_bs = d.shape[0]
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, local_bs, feats)
			decoder1 = getattr(model, "transformer_decoder1", None)
			decoder2 = getattr(model, "transformer_decoder2", None)

			# Step A (L1): freeze Decoder2, update (E, Decoder1) by minimizing L1
			_set_requires_grad(decoder2, False)
			_set_requires_grad(decoder1, True)
			z = model(window, elem)

			if isinstance(z, tuple) and len(z) == 3:
				O1, Oh2, O2 = z[0], z[1], z[2]
				err_O1 = l(O1, elem)    # ||O1 - W||^2
				err_Oh2 = l(Oh2, elem) # ||Ô2 - W||^2
				err_O2 = l(O2, elem)    # ||O2 - W||^2

				L1 = torch.mean(eta * err_O1 + (1 - eta) * err_Oh2)
				optimizer.zero_grad()
				L1.backward()
				optimizer.step()
				l1s.append(L1.detach().item())

				# Step B (L2): freeze Decoder1, update (E, Decoder2) by minimizing L2
				_set_requires_grad(decoder2, True)
				_set_requires_grad(decoder1, False)
				z2 = model(window, elem)

				O1b, Oh2b, O2b = z2[0], z2[1], z2[2]
				err_Oh2b = l(Oh2b, elem)
				err_O2b = l(O2b, elem)
				L2 = torch.mean(eta * err_O2b - (1 - eta) * err_Oh2b)

				optimizer.zero_grad()
				L2.backward()
				optimizer.step()
				l2s.append(L2.detach().item())

				# Restore trainability
				_set_requires_grad(decoder2, True)
				_set_requires_grad(decoder1, True)

			else:
				# Fallback: keep the previous behavior if the model doesn't match the expected output format.
				# (This path is primarily for other model variants.)
				_set_requires_grad(decoder2, True)
				_set_requires_grad(decoder1, True)
				loss_elem = l(z, elem) if not isinstance(z, tuple) else (eta * l(z[0], elem) + (1 - eta) * l(z[1], elem))
				l1s.append(torch.mean(loss_elem).item())
				optimizer.zero_grad()
				torch.mean(loss_elem).backward()
				optimizer.step()

		scheduler.step()
		# Report both losses for easier debugging.
		if len(l2s) > 0:
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
		else:
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
		return np.mean(l1s), optimizer.param_groups[0]['lr']
	else:
		# Paper-align: inference anomaly score should combine Phase1/Phase2 reconstruction errors.
		# Keep original behavior by toggling the flag below.
		use_paper_score = True
		for d, _ in dataloader:
			window = d.permute(1, 0, 2)
			elem = window[-1, :, :].view(1, bs, feats)
			z = model(window, elem)

			if isinstance(z, tuple):
				x1, x2 = z[0], z[1]
				# loss shape: (1, bs, feats) -> take [0] => (bs, feats)
				if use_paper_score:
					err1 = l(x1, elem)[0]
					err2 = l(x2, elem)[0]
					loss = 0.5 * err1 + 0.5 * err2
				else:
					loss = l(x2, elem)[0]
				# Keep plotting/prediction output consistent with original code.
				z_plot = x2
			else:
				z_plot = z
				loss = l(z_plot, elem)[0]

		return loss.detach().numpy(), z_plot.detach().numpy()[0]

if __name__ == '__main__':
	train_loader, test_loader, labels = load_dataset(args.dataset)
	model, optimizer, scheduler, epoch, accuracy_list = load_model(args.model, labels.shape[1])

	## Prepare data
	trainD, testD = next(iter(train_loader)), next(iter(test_loader))
	lable = next(iter(labels))
	trainO, testO = trainD, testD
	trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)

	### Training phase
	if not args.test:
		print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
		num_epochs = 50 if args.dataset in ['UCR', 'MBA' , 'SMAP' ] else 15
		e = epoch + 1; start = time()
		for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
			lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
			accuracy_list.append((lossT, lr))
		print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
		save_model(model, optimizer, scheduler, e, accuracy_list)
		plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

	### Testing phase
	torch.zero_grad = True
	model.eval()
	print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
	loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
	### Scores
	df = pd.DataFrame()
	lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)

	### lt is the train anomaly score, l is the test anomaly score. ls is the labels
	### pred is result about 0/1,and the result is the Evaluation Criteria(f1...)
	pot_thresholds = []
	pot_pred_list = []
	for i in range(loss.shape[1]):
		# pprint(loss.shape[1])
		lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
		result, pred = pot_eval(lt, l, ls)
		preds.append(pred)
		df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
		if 'threshold' in result:
			pot_thresholds.append(result['threshold'])
		else:
			pot_thresholds.append(None)
		pot_pred_list.append(pred)
	lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
	labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
	result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
	result.update(hit_att(loss, labels))
	result.update(ndcg(loss, labels))
	pprint(result)

	### Plot curves with POT overlays
	if not args.test:
		try:
			pot_preds = np.stack(pot_pred_list, axis=1) if len(pot_pred_list) else None
		except Exception:
			pot_preds = None
		testO_plot = torch.roll(testO, 1, 0)
		plotter(
			f'{args.model}_{args.dataset}',
			testO_plot, y_pred, loss, labels,
			pot_thresholds=pot_thresholds if len(pot_thresholds) else None,
			pot_preds=pot_preds
		)
