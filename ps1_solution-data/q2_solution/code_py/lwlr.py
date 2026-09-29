import torch

def lwlr_torch(X_train, y_train, x, tau, reg1=1e-4, reg2=2e-6, tol=1e-6):
    """
    Locally weighted logistic regression X_train: (m, n) tensor - ncludes the bias term already y_train: (m, 1) or (m,) tensor with 0/1 labels.
    x:(n,) or (n,1) tensor - query point
    tau: float - badnwidth
    """

    m,n = X_train.shape

    if y_train.dim() == 1:
        y_train = y_train.view(-1,1)
    if x.dim() == 2:
        x = x.view(-1)

    theta = torch.zeros(n, 1, dtype=X_train.dtype, device=X_train.device)

    # compute the weights: w = exp(-||X_train - x||^2 / (2*tau))
    # This is the Gaussian Kernel
    diff = X_train - x # broadcasting: (m,n) - (n) -> (m,n)
    w = torch.exp(-torch.sum(diff**2, dim=1) / (2 * tau)) # (m,)
    w = w.view(-1,1) #(m,b) for broadcasting

    # 2. Newton's Method
    g = torch.ones(n,1, dtype=X_train.dtype, device=X_train.device)

    while torch.norm(g) > tol:
        # h = sigmoid(X * theta)
        h = torch.sigmoid(X_train @ theta) # (m,1)

        #gradient: X '*(w * (y - h)) - reg * theta
        g = X_train.T@(w * (y_train - h)) - reg1 * theta

        # Hessian: -X' * diag(w*h*(1-h)) * X - reg*I
        wh = w * h * (1 - h) # (m,b)

        # Effiicent way: X.T @ (wh * X) isntad od forming diag
        H = -X_train.T@(wh * X_train) - reg2 * torch.eye(n, dtype=X_train.dtype, device=X_train.device)

        # theta = theta - H \g -> solve(H,g)
        # This is the MATLAB H\g
        
        delta = torch.linalg.solve(H,g)
        theta = theta - delta

        # Return predicted y
        y_pred = 1.0 if (x @ theta.view(-1) > 0) else 0.0
        return y_pred, theta