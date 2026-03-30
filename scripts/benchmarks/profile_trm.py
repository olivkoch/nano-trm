import click
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from src.nn.data.sudoku_datamodule import SudokuDataModule
from src.nn.models.trm import TRMModule

def profile_inference(model, batch):

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    carry = model.initial_carry(batch)

    # warmup 
    with torch.no_grad():
        for _ in range(3):
            model.forward(carry, batch)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/trm_profile'),
        record_shapes=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(5):  # Multiple iterations to clear warmup
                with record_function("trm_inference_step"):
                    model.forward(carry, batch)
                prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

@click.command()
@click.option('--data_path', default='data/sudoku', help='Path to the dataset')
@click.option('--batch_size', default=512, help='Batch size for profiling')
@click.option('--compile', is_flag=True, help='Whether to use torch.compile')
def main(data_path, batch_size, compile):
    dm = SudokuDataModule(
        data_dir=data_path,
        batch_size=batch_size,
        num_workers=0,)

    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    model = TRMModule(
        hidden_size=128,
        num_layers=2,
        puzzle_emb_dim=128,
        puzzle_emb_len=4,
        N_supervision=3,
        num_puzzles=dm.num_puzzles,
        batch_size=dm.batch_size,
        pad_value=dm.pad_value,
        max_grid_size=dm.max_grid_size,
        vocab_size=dm.vocab_size,
        seq_len=dm.max_grid_size * dm.max_grid_size,
    )

    if compile:
        print("Compiling model...")
        model = torch.compile(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    profile_inference(model, batch)

if __name__ == "__main__":
    main()