import torch


def recurrent_cell_init(worker_num, hidden_state_size, layer_type, device) -> tuple:
    """Initializes the recurrent cell states (hxs, cxs) as zeros.
    Args:
        task_num {int} -- Number of tasks.
        worker_per_task {int} -- Number of workers per task.
        hidden_state_size {int} -- Size of the hidden state.
        layer_type {str} -- Type of the recurrent layer.
        device {torch.device} -- Device to use.
    Returns:
        {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                    cell states are returned using initial values.
    """
    if layer_type == "gru":
        return torch.zeros(
            (1, worker_num, hidden_state_size),
            dtype=torch.float32,
            device=device,
        )

    elif layer_type == "lstm":
        return (
            torch.zeros(
                (1, worker_num, hidden_state_size),
                dtype=torch.float32,
                device=device,
            ),
            torch.zeros(
                (1, worker_num, hidden_state_size),
                dtype=torch.float32,
                device=device,
            ),
        )
    else:
        raise NotImplementedError(layer_type)
