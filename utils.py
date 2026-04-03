import numpy as np 

def binary_loss(Z, Y):
    # Retourne 0 si Z >= Y pour tous les pixels, sinon 1 
    return 0 if np.all(Z >= Y) else 1
    
def threshold_binary_loss(Z, Y, tau=0.95):
    # Somme sur i,j,k pour compter les succès de couverture 
    ratio = np.sum(Z * Y) / np.sum(Y)
    return 0 if ratio >= tau else 1
    
def miscoverage_loss(Z, Y):
    # Perte de non-couverture directe 
    return 1 - (np.sum(Z * Y) / np.sum(Y))

def dichotomie(f, alpha, max_iter=100, tol=1e-7):
    low = 0.0
    high = 1.0
    best_l = 1.0
    
    for _ in range(max_iter):
        mid = (low + high) / 2
        current_risk = f(mid)
        
        # Si le risque est inférieur à alpha, on tente de réduire l'imprécision (diminuer lambda)
        # Sinon, on doit augmenter lambda pour couvrir plus de classes
        if current_risk <= alpha:
            best_l = mid
            high = mid
        else:
            low = mid
            
        if abs(high - low) < tol:
            break
            
    return best_l

def thresholding(f_X, l):
    # f_X shape: (K, H, W)
    mask = (f_X >= (1 - l)).astype(int)
    
    # Optionnel mais recommandé : Inclure toujours la classe top-1 
    top1 = np.zeros_like(f_X)
    max_indices = np.argmax(f_X, axis=0)
    for k in range(f_X.shape[0]):
        top1[k, :, :] = (max_indices == k)
        
    return np.maximum(mask, top1)

def threshold_CRC(alpha, proba_list, Y_list, loss_func, B=1):
    n = len(proba_list)
    
    def f(l):
        # 1. Appliquer le seuillage à tout le dataset de calibration 
        Z_list = [thresholding(p, l) for p in proba_list]
        # 2. Calculer le risque moyen 
        risk_empirique = np.mean([loss_func(z, y) for z, y in zip(Z_list, Y_list)])
        # 3. Appliquer la formule de la borne supérieure 
        return (n / (n + 1)) * risk_empirique + (B / (n + 1))
    
    # Trouver le plus petit lambda satisfaisant la condition 
    return dichotomie(f, alpha)

