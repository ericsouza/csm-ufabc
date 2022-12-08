# ESTI019 - CSM - Lab4

# Lab4 - Codificação de Imagem com DWT 

A. Objetivos:


1.   Efetuar a Codificação de Imagem e a Decodificação por DWT e IDWT
2.   Testar funções de Codificação Multinível
3.   Verificar a taxa de compressão só com a Componente de Aproximação


```python
import numpy as np
import cv2 as cv
import pywt
import pywt.data
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
```

Codificação de Luminância (P&B) com DWT para a Pimentas


```python
img = mpimg.imread('files/lab4/peppers.png')

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)


coefs2 = pywt.dwt2(img_gray,'haar', mode='periodization')  #1 nível de DWT
(cA, (cH, cV, cD)) = coefs2 #Separando os coeficientes
imgr = pywt.idwt2(coefs2, 'haar', mode = 'periodization')  #1 nível de IDWT

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(cA,'gray'); plt.title("CA - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV,'gray'); plt.title("CV - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH,'gray'); plt.title("CH - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD,'gray'); plt.title("CD - Bordas Diagonais")

```




    Text(0.5, 1.0, 'CD - Bordas Diagonais')




    
![png](output_5_1.png)
    


Cálculo do Erro Quadrático Médio (MSE) e da Relação Sinal Ruído de Pico (PSNR)

A MSE é obtida calculando somando-se o erro quadrático de reconstrução pixel a pixel entre a Imagem Original (O) da Reconstruída (R) e normalizando pela dimensão (LxA) da imagem:

$ MSE = \frac{1}{LA}{\sum_{i=0}^L}{\sum_{j=0}^A [O(i,j) - R(i,j)]^2}$


```python
# Calculo da MSE P&B
A, L, Camadas = img.shape
dif = img_gray - imgr
MSE_gray = np.sum(np.matmul(dif,np.transpose(dif)))/(A*L)
print("MSE_Y = {:.2e}".format(MSE_gray))
```

    MSE_Y = 1.62e-13


A SNR de pico (PSNR) é definida para cada plano componente da imagem como:

$ PSNR = 10.log_{10} \left( \frac{{MAX_I}^2}{MSE} \right) $

sendo $MAX_I$ o valor máximo do pixel, que para 8 bits equivale a 255, logo:

$ PSNR = 20.log_{10}(255) - 10.log_{10} (MSE) $

OBS.: Para uma imagem RGB, $ MSE = MSE_R + MSE_G + MSE_B $, sendo similar definiação para YCrCb e HSV


```python
PSNR_Y = 20*np.log10(255) - 10*np.log10(MSE_gray)
print("PSNR_Luma = {:.2f} dB".format(PSNR_Y))
plt.figure(figsize=(20,10))
infograf = "Imagem Reconstruída de Luminância (Y) com PSNR = " + str(np.uint8(PSNR_Y)) + ' dB'
plt.subplot(1,2,1); plt.imshow(img_gray,'gray'); plt.title("Imagem Original P&B")
plt.subplot(1,2,2); plt.imshow(imgr,'gray'); plt.title(infograf)
```

    PSNR_Luma = 176.03 dB





    Text(0.5, 1.0, 'Imagem Reconstruída de Luminância (Y) com PSNR = 176 dB')




    
![png](output_10_2.png)
    


Teste das Funções de Multiresolução wavedec2() e waverec2()


```python
C = pywt.wavedec2(img_gray,'haar', mode = 'symmetric', level=2) # Dois níveis de decomposição DWT
imgr2 = pywt.waverec2(C, 'haar', mode = 'symmetric') # Dois níveis de IDWT

# Para extrair os coeficientes de cada nível
cA2 = C[0]  # Coeficientes de Aproximação nível 2
(cH1, cV1, cD1) = C[-1] # Coeficientes de Detalhes nível 1
(cH2, cV2, cD2) = C[-2] # Coeficientes de Detalhes nível 2

# Imagem Original
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Plot dos coeficientes do nível 2
plt.figure(figsize=(5,5))
plt.subplot(2,2,1)
plt.imshow(cA2, 'gray'); plt.title('Ap. N2: CA2')
plt.subplot(2,2,2)
plt.imshow(cV2, 'gray'); plt.title('B. V. N2: CV2')
plt.subplot(2,2,3)
plt.imshow(cH2, 'gray'); plt.title('B. H. N2: CH2')
plt.subplot(2,2,4)
plt.imshow(cD2, 'gray'); plt.title('B. D. N2: CD2')

# Plot Original e Reconstrução
plt.figure(figsize=(20,10))
plt.subplot(1,2,1); plt.imshow(img_gray,'gray'); plt.title('Imagem Original')
plt.subplot(1,2,2); plt.imshow(imgr2,'gray'); plt.title('Imagem Reconstruída')

```




    Text(0.5, 1.0, 'Imagem Reconstruída')




    
![png](output_12_1.png)
    



    
![png](output_12_2.png)
    


Efetuar uma "Montagem" com wavedec2() e wavedecn() 

1º Nível


```python
CV1 = cV1.copy()
CH1 = cH1.copy()
CD1 = cD1.copy()

```

2º Nível


```python
CA2 = cA2.copy()
CH2 = cH2.copy()
CV2 = cV2.copy()
CD2 = cD2.copy()
# Matriz Final Completa
CA1 = np.bmat([[CA2,CV2],[CH2,CD2]])
CC = np.bmat([[CA1,CV1],[CH1,CD1]])
```


```python
plt.figure(figsize=(20,20))
plt.imshow(CC,'gray')
plt.title('Codificação de Imagem em multinível com função wavedec2()')
```




    Text(0.5, 1.0, 'Codificação de Imagem em multinível com função wavedec2()')




    
![png](output_18_1.png)
    


Reconstrução de Imagem Colorida


```python
# Imagem Original

plt.figure(figsize=(20,20))
plt.imshow(img); plt.title("Imagem Original")

# Codificação por planos de cores
# Plano Vermelho
coefs_R = pywt.dwt2(img[:,:,0],'haar', mode='periodization')  #1 nível de DWT R
(cA_R, (cH_R, cV_R, cD_R)) = coefs_R #Separando os coeficientes
cr_R = pywt.idwt2(coefs_R, 'haar', mode = 'periodization')  #1 nível de IDWT R

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_R,'Reds_r'); plt.title("CA_Red - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_R,'Reds_r'); plt.title("CV_Red - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_R,'Reds_r'); plt.title("CH_Red - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_R,'Reds_r'); plt.title("CD_Red - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_R, 'Reds_r'); plt.title("Imagem Reconstruída Red")

```




    Text(0.5, 1.0, 'Imagem Reconstruída Red')




    
![png](output_20_1.png)
    



    
![png](output_20_2.png)
    



    
![png](output_20_3.png)
    



```python
# Plano Verde
coefs_G = pywt.dwt2(img[:,:,1],'haar', mode='periodization')  #1 nível de DWT G
(cA_G, (cH_G, cV_G, cD_G)) = coefs_G #Separando os coeficientes
cr_G = pywt.idwt2(coefs_G, 'haar', mode = 'periodization')  #1 nível de IDWT G

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_G,'Greens_r'); plt.title("CA_Green - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_G,'Greens_r'); plt.title("CV_Green - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_G,'Greens_r'); plt.title("CH_Green - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_G,'Greens_r'); plt.title("CD_Green - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_G, 'Greens_r'); plt.title("Imagem Reconstruída Green")
```




    Text(0.5, 1.0, 'Imagem Reconstruída Green')




    
![png](output_21_1.png)
    



    
![png](output_21_2.png)
    



```python
# Plano Azul
coefs_B = pywt.dwt2(img[:,:,2],'haar', mode='periodization')  #1 nível de DWT B
(cA_B, (cH_B, cV_B, cD_B)) = coefs_B #Separando os coeficientes
cr_B = pywt.idwt2(coefs_B, 'haar', mode = 'periodization')  #1 nível de IDWT B

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_B,'Blues_r'); plt.title("CA_Blue - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_B,'Blues_r'); plt.title("CV_Blue - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_B,'Blues_r'); plt.title("CH_Blue - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_B,'Blues_r'); plt.title("CD_Blue - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_B, 'Blues_r'); plt.title("Imagem Reconstruída Blue")
```




    Text(0.5, 1.0, 'Imagem Reconstruída Blue')




    
![png](output_22_1.png)
    



    
![png](output_22_2.png)
    



```python
# Reconstrução Nível 1 Colorida 
A1, L1 = cA_R.shape
imgrec1 = np.zeros((A1,L1,3))
imgrec1[:,:,0] = cA_R.copy()
imgrec1[:,:,1] = cA_G.copy()
imgrec1[:,:,2] = cA_B.copy()
plt.figure(figsize=(10,10))
plt.imshow(imgrec1); plt.title("Imagem Reconstruída DWT/IDWT Nível 1")

```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).





    Text(0.5, 1.0, 'Imagem Reconstruída DWT/IDWT Nível 1')




    
![png](output_23_2.png)
    


Salvando as Aproximações e depois fazendo download dos arquivos, calcular a taxa de compressão com o original


```python
# obtenha aproximação de NÍVEL 1 e converte para inteiro
C1 = pywt.wavedec2(img_gray,'haar', mode = 'symmetric', level=1) # Um nível de decomposição DWT
CA1_ = 255 * C1[0] / np.abs(C1[0]).max()
CA1_ = CA1_.astype(int)

# aproximação de NÍVEL 2 - ja´obtido no item (D) e converte para inteiro
CA2_ = 255 * cA2 / np.abs(cA2).max()
CA2_ = CA2_.astype(int)

# Salva no drive
cv.imwrite('files/lab4/peppers_DWT_N1_Y.bmp', CA1_) # Aproximação Nível 1 só Y
cv.imwrite('files/lab4/peppers_DWT_N2_Y.bmp', CA2_) # Aproximação Nível 2 só Y

```




    True




```python
import os
tamanho = os.path.getsize('files/lab4/peppers_DWT_N1_Y.bmp')
print(tamanho)
```

    50230


Gravando o Arquivo Codificado DWT/IDWT nível 1 Colorido, calcular a taxa de compressão com o original


```python
# Aproximação Nível 1 Colorida 
# converte RGB para BGR e converte para inteiro
imgrec1_ = np.zeros((A1,L1,3))
imgrec1_[:,:,0] = imgrec1[:,:,2] 
imgrec1_[:,:,1] = imgrec1[:,:,1] 
imgrec1_[:,:,2] = imgrec1[:,:,0] 
imgrec1_ = ( 255 * imgrec1_ / np.abs(imgrec1_).max() ).astype(int)

# Salva no drive
cv.imwrite('files/lab4/peppers_DWT_N1_colorida.bmp', imgrec1_) # Gravando Aproximação Nível 1 Colorida
```




    True



Reconstrução da Imagem colorida e Cálculo da MSE de cada plano de cor e da PSNR total


```python
# Reconstrução Colorida Original
A,L = imgr2.shape
imgrec = np.zeros((A,L,3))
imgrec[:,:,0] = cr_R.copy()
imgrec[:,:,1] = cr_G.copy()
imgrec[:,:,2] = cr_B.copy()

# Calculo do MSE colorida
dif2 = img - imgrec
MSE_R = np.sum(np.matmul(dif2[:,:,0],np.transpose(dif2[:,:,0])))/(A*L) # Erro Quadrático Médio plano R
MSE_G = np.sum(np.matmul(dif2[:,:,1],np.transpose(dif2[:,:,1])))/(A*L) # Erro Quadrático Médio plano G
MSE_B = np.sum(np.matmul(dif2[:,:,2],np.transpose(dif2[:,:,2])))/(A*L) # Erro Quadrático Médio plano B
print("MSE_Red= {:.2e}".format(MSE_R), " MSE_Green= {:.2e}".format(MSE_G), " MSE_Blue= {:.2e}".format(MSE_B))

# Cálculo da SNR de pico colorida (PSNR), 3 camadas R, G e B
PSNR = 20*np.log10(255) - 10*np.log10(MSE_R + MSE_G + MSE_B)
print("PSNR total = {:.2f} dB".format(PSNR))

plt.figure(figsize=(20,20))
infograf2 = "Imagem Reconstruída com PSNR total = " + str(np.uint8(PSNR)) + ' dB'
plt.imshow(imgrec); plt.title(infograf2)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    MSE_Red= 4.28e-13  MSE_Green= 1.12e-13  MSE_Blue= 8.70e-14
    PSNR total = 170.16 dB





    Text(0.5, 1.0, 'Imagem Reconstruída com PSNR total = 170 dB')




    
![png](output_30_3.png)
    


# I. Repetindo etapas com cada componente

##### Codificação de Luminância (P&B) com DWT para a Pimentas

### Carregando imagens


```python
img_files = ["1.jpg","2.jpg","3.jpg","4.jpg"]
```


```python
img = []
img_gray = []

for x in img_files:

    img_read = mpimg.imread('files/lab4/'+x)
    img.append(img_read)
    img_gray.append(cv.cvtColor(img_read, cv.COLOR_RGB2GRAY))
imgr = []
for x in range(len(img_files)):
    coefs2 = pywt.dwt2(img_gray[x],'haar', mode='periodization')  #1 nível de DWT
    (cA, (cH, cV, cD)) = coefs2 #Separando os coeficientes
    imgr.append(pywt.idwt2(coefs2, 'haar', mode = 'periodization'))  #1 nível de IDWT

    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(cA,'gray'); plt.title("CA - Aproximação - Imagem "+str(x+1))
    plt.subplot(2,2,2)
    plt.imshow(cV,'gray'); plt.title("CV - Bordas Verticais - Imagem "+str(x+1))
    plt.subplot(2,2,3)
    plt.imshow(cH,'gray'); plt.title("CH - Bordas Horizontais- Imagem "+str(x+1))
    plt.subplot(2,2,4)
    plt.imshow(cD,'gray'); plt.title("CD - Bordas Diagonais- Imagem "+str(x+1))
```


    
![png](output_35_0.png)
    



    
![png](output_35_1.png)
    



    
![png](output_35_2.png)
    



    
![png](output_35_3.png)
    


### Cálculo do Erro Quadrático Médio (MSE) e da Relação Sinal Ruído de Pico (PSNR)


```python
# Calculo da MSE P&B
MSE_gray = []
for x in range(len(img)):
    A, L, Camadas = img[x].shape
    dif = img_gray[x] - imgr[x]
    MSE_gray.append(np.sum(np.matmul(dif,np.transpose(dif)))/(A*L))
    print("MSE_Y = {:.2e} - Imagem_".format(MSE_gray[x])+str(x+1))
```

    MSE_Y = 5.30e-24 - Imagem_1
    MSE_Y = 6.24e-24 - Imagem_2
    MSE_Y = 5.51e-24 - Imagem_3
    MSE_Y = 4.72e-24 - Imagem_4



```python
for x in range(len(img)):
    PSNR_Y = 20*np.log10(255) - 10*np.log10(MSE_gray[x])
    print("PSNR_Luma = {:.2f} dB".format(PSNR_Y)+" Imagem_"+str(x))
    plt.figure(figsize=(20,10))
    infograf = "Imagem Reconstruída de Luminância (Y) com PSNR = " + str(np.uint8(PSNR_Y)) + ' dB'+"_Imagem_"+str(x)
    plt.subplot(1,2,1); plt.imshow(img_gray[x],'gray'); plt.title("Imagem Original P&B - Imagem_"+str(x))
    plt.subplot(1,2,2); plt.imshow(imgr[x],'gray'); plt.title(infograf)
```

    PSNR_Luma = 280.88 dB Imagem_0
    PSNR_Luma = 280.18 dB Imagem_1
    PSNR_Luma = 280.72 dB Imagem_2
    PSNR_Luma = 281.39 dB Imagem_3



    
![png](output_38_1.png)
    



    
![png](output_38_2.png)
    



    
![png](output_38_3.png)
    



    
![png](output_38_4.png)
    


### Teste das Funções de Multiresolução wavedec2() e waverec2()


```python
C = pywt.wavedec2(img_gray[x],'haar', mode = 'symmetric', level=2)
```


```python
imgr2 = []
C = []
for x in range(len(img)):
    C.append(pywt.wavedec2(img_gray[x],'haar', mode = 'symmetric', level=2)) # Dois níveis de decomposição DWT
    imgr2.append(pywt.waverec2(C[x], 'haar', mode = 'symmetric')) # Dois níveis de IDWT

    # Para extrair os coeficientes de cada nível
    cA2 = C[x][0]  # Coeficientes de Aproximação nível 2
    (cH1, cV1, cD1) = C[x][-1] # Coeficientes de Detalhes nível 1
    (cH2, cV2, cD2) = C[x][-2] # Coeficientes de Detalhes nível 2

    # Imagem Original
    # img_gray[x] = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Plot dos coeficientes do nível 2
    plt.figure(figsize=(5,5))
    plt.subplot(2,2,1)
    plt.imshow(cA2, 'gray'); plt.title('Ap. N2: CA2')
    plt.subplot(2,2,2)
    plt.imshow(cV2, 'gray'); plt.title('B. V. N2: CV2')
    plt.subplot(2,2,3)
    plt.imshow(cH2, 'gray'); plt.title('B. H. N2: CH2')
    plt.subplot(2,2,4)
    plt.imshow(cD2, 'gray'); plt.title('B. D. N2: CD2')

    # Plot Original e Reconstrução
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1); plt.imshow(img_gray[x],'gray'); plt.title('Imagem Original')
    plt.subplot(1,2,2); plt.imshow(imgr2[x],'gray'); plt.title('Imagem Reconstruída')
```


    
![png](output_41_0.png)
    



    
![png](output_41_1.png)
    



    
![png](output_41_2.png)
    



    
![png](output_41_3.png)
    



    
![png](output_41_4.png)
    



    
![png](output_41_5.png)
    



    
![png](output_41_6.png)
    



    
![png](output_41_7.png)
    


### Efetuar uma "Montagem" com wavedec2() e wavedecn() 


```python
for x in range(len(C)):

    cA2 = C[x][0]  # Coeficientes de Aproximação nível 2
    (cH1, cV1, cD1) = C[x][-1] # Coeficientes de Detalhes nível 1
    (cH2, cV2, cD2) = C[x][-2] # Coeficientes de Detalhes nível 2

    #Primeiro nível
    CV1 = cV1.copy()
    CH1 = cH1.copy()
    CD1 = cD1.copy()

    #Segundo nível
    CA2 = cA2.copy()
    CH2 = cH2.copy()
    CV2 = cV2.copy()
    CD2 = cD2.copy()
    # Matriz Final Completa
    CA1 = np.bmat([[CA2,CV2],[CH2,CD2]])
    CC = np.bmat([[CA1,CV1],[CH1,CD1]])


    plt.figure(figsize=(20,20))
    plt.imshow(CC,'gray')
    plt.title('Codificação de Imagem em multinível com função wavedec2() - Imagem '+str(x))
```


    
![png](output_43_0.png)
    



    
![png](output_43_1.png)
    



    
![png](output_43_2.png)
    



    
![png](output_43_3.png)
    


### Reconstrução de Imagem Colorida

#### Vermelho


```python
CA_R = []
CR_R = []
for x in range(len(img)):

    # Imagem Original

    plt.figure(figsize=(20,20))
    plt.imshow(img[x]); plt.title("Imagem Original - "+str(x))

    # Codificação por planos de cores
    # Plano Vermelho
    coefs_R = pywt.dwt2(img[x][:,:,0],'haar', mode='periodization')  #1 nível de DWT R
    (cA_R, (cH_R, cV_R, cD_R)) = coefs_R #Separando os coeficientes
    cr_R = pywt.idwt2(coefs_R, 'haar', mode = 'periodization')  #1 nível de IDWT R
    CA_R.append(cA_R)
    CR_R.append(cr_R)

    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.imshow(cA_R,'Reds_r'); plt.title("CA_Red - Aproximação")
    plt.subplot(2,2,2)
    plt.imshow(cV_R,'Reds_r'); plt.title("CV_Red - Bordas Verticais")
    plt.subplot(2,2,3)
    plt.imshow(cH_R,'Reds_r'); plt.title("CH_Red - Bordas Horizontais")
    plt.subplot(2,2,4)
    plt.imshow(cD_R,'Reds_r'); plt.title("CD_Red - Bordas Diagonais")

    plt.figure(figsize=(20,20))
    plt.imshow(cr_R, 'Reds_r'); plt.title("Imagem Reconstruída Red")
```


    
![png](output_46_0.png)
    



    
![png](output_46_1.png)
    



    
![png](output_46_2.png)
    



    
![png](output_46_3.png)
    



    
![png](output_46_4.png)
    



    
![png](output_46_5.png)
    



    
![png](output_46_6.png)
    



    
![png](output_46_7.png)
    



    
![png](output_46_8.png)
    



    
![png](output_46_9.png)
    



    
![png](output_46_10.png)
    



    
![png](output_46_11.png)
    


#### Verde


```python
# Plano Verde
CA_G = []
CR_G = []
# Imagem Original
for x in range(len(img)):
    plt.figure(figsize=(20,20))
    plt.imshow(img[x]); plt.title("Imagem Original - "+str(x))

    coefs_G = pywt.dwt2(img[x][:,:,1],'haar', mode='periodization')  #1 nível de DWT G
    (cA_G, (cH_G, cV_G, cD_G)) = coefs_G #Separando os coeficientes
    cr_G = pywt.idwt2(coefs_G, 'haar', mode = 'periodization')  #1 nível de IDWT G

    CA_G.append(cA_G)
    CR_G.append(cr_G)

    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.imshow(cA_G,'Greens_r'); plt.title("CA_Green - Aproximação")
    plt.subplot(2,2,2)
    plt.imshow(cV_G,'Greens_r'); plt.title("CV_Green - Bordas Verticais")
    plt.subplot(2,2,3)
    plt.imshow(cH_G,'Greens_r'); plt.title("CH_Green - Bordas Horizontais")
    plt.subplot(2,2,4)
    plt.imshow(cD_G,'Greens_r'); plt.title("CD_Green - Bordas Diagonais")

    plt.figure(figsize=(20,20))
    plt.imshow(cr_G, 'Greens_r'); plt.title("Imagem Reconstruída Green")
```


    
![png](output_48_0.png)
    



    
![png](output_48_1.png)
    



    
![png](output_48_2.png)
    



    
![png](output_48_3.png)
    



    
![png](output_48_4.png)
    



    
![png](output_48_5.png)
    



    
![png](output_48_6.png)
    



    
![png](output_48_7.png)
    



    
![png](output_48_8.png)
    



    
![png](output_48_9.png)
    



    
![png](output_48_10.png)
    



    
![png](output_48_11.png)
    


#### Azul


```python
# Plano Azul
CA_B = []
CR_B = []
for x in range(len(img)):
    plt.figure(figsize=(20,20))
    plt.imshow(img[x]); plt.title("Imagem Original - "+str(x))

    coefs_B = pywt.dwt2(img[x][:,:,2],'haar', mode='periodization')  #1 nível de DWT B
    (cA_B, (cH_B, cV_B, cD_B)) = coefs_B #Separando os coeficientes
    cr_B = pywt.idwt2(coefs_B, 'haar', mode = 'periodization')  #1 nível de IDWT B
    CA_B.append(cA_B)
    CR_B.append(cr_B)
    plt.figure(figsize=(20,20))
    plt.subplot(2,2,1)
    plt.imshow(cA_B,'Blues_r'); plt.title("CA_Blue - Aproximação")
    plt.subplot(2,2,2)
    plt.imshow(cV_B,'Blues_r'); plt.title("CV_Blue - Bordas Verticais")
    plt.subplot(2,2,3)
    plt.imshow(cH_B,'Blues_r'); plt.title("CH_Blue - Bordas Horizontais")
    plt.subplot(2,2,4)
    plt.imshow(cD_B,'Blues_r'); plt.title("CD_Blue - Bordas Diagonais")

    plt.figure(figsize=(20,20))
    plt.imshow(cr_B, 'Blues_r'); plt.title("Imagem Reconstruída Blue")
```


    
![png](output_50_0.png)
    



    
![png](output_50_1.png)
    



    
![png](output_50_2.png)
    



    
![png](output_50_3.png)
    



    
![png](output_50_4.png)
    



    
![png](output_50_5.png)
    



    
![png](output_50_6.png)
    



    
![png](output_50_7.png)
    



    
![png](output_50_8.png)
    



    
![png](output_50_9.png)
    



    
![png](output_50_10.png)
    



    
![png](output_50_11.png)
    


#### Reconstrução Nível 1 Colorida 


```python
# Reconstrução Nível 1 Colorida
imgrec1_list = []
for x in range(len(img)):
    cA_R = CA_R[x]
    cA_G = CA_G[x]
    cA_B = CA_B[x]

    A1, L1 = cA_R.shape
    imgrec1 = np.zeros((A1,L1,3))
    imgrec1[:,:,0] = cA_R.copy()
    imgrec1[:,:,1] = cA_G.copy()
    imgrec1[:,:,2] = cA_B.copy()
    imgrec1_list.append(imgrec1)
    plt.figure(figsize=(10,10))
    plt.imshow(imgrec1); plt.title("Imagem Reconstruída DWT/IDWT Nível 1")
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).



    
![png](output_52_1.png)
    



    
![png](output_52_2.png)
    



    
![png](output_52_3.png)
    



    
![png](output_52_4.png)
    


### Salvando as Aproximações e calculando a taxa de compressão com o original


```python
# obtenha aproximação de NÍVEL 1 e converte para inteiro

for x in range(len(img)):
    print(x)
    C1 = pywt.wavedec2(img_gray[x],'haar', mode = 'symmetric', level=1) # Um nível de decomposição DWT
    CA1_ = 255 * C1[0] / np.abs(C1[0]).max()
    CA1_ = CA1_.astype(int)

    # aproximação de NÍVEL 2 - ja´obtido no item (D) e converte para inteiro
    CA2_ = 255 * cA2 / np.abs(cA2).max()
    CA2_ = CA2_.astype(int)

    # Salva no drive
    cv.imwrite('files/lab4/'+str(x)+'_DWT_N1_Y.bmp', CA1_) # Aproximação Nível 1 só Y
    cv.imwrite('files/lab4/'+str(x)+'_DWT_N2_Y.bmp', CA2_) # Aproximação Nível 2 só Y
```

    0
    1
    2
    3


### Gravando o Arquivo Codificado DWT/IDWT nível 1 Colorido, calcular a taxa de compressão com o original


```python
# Aproximação Nível 1 Colorida 
# converte RGB para BGR e converte para inteiro
# Reconstrução Nível 1 Colorida 
for x in range(len(img)):
    A1, L1 = CA_R[x].shape
    imgrec1_ = np.zeros((A1,L1,3))
    imgrec1_[:,:,0] = imgrec1_list[x][:,:,2] 
    imgrec1_[:,:,1] = imgrec1_list[x][:,:,1] 
    imgrec1_[:,:,2] = imgrec1_list[x][:,:,0] 
    imgrec1_ = ( 255 * imgrec1_ / np.abs(imgrec1_).max() ).astype(int)  
    # Salva no drive
    cv.imwrite('files/lab4/'+str(x)+'_DWT_N1_colorida.bmp', imgrec1_) # Gravando Aproximação Nível 1 Colorida
```

### Reconstrução da Imagem colorida e Cálculo da MSE de cada plano de cor e da PSNR total


```python
# Reconstrução Colorida Original
for x in range(len(imgr2)):
    A,L = imgr2[x].shape
    imgrec = np.zeros((A,L,3))
    imgrec[:,:,0] = CR_R[x].copy()
    imgrec[:,:,1] = CR_G[x].copy()
    imgrec[:,:,2] = CR_B[x].copy()

    # # Calculo do MSE colorida

    dif2 = img[x] - imgrec
    MSE_R = np.sum(np.matmul(dif2[:,:,0],np.transpose(dif2[:,:,0])))/(A*L) # Erro Quadrático Médio plano R
    MSE_G = np.sum(np.matmul(dif2[:,:,1],np.transpose(dif2[:,:,1])))/(A*L) # Erro Quadrático Médio plano G
    MSE_B = np.sum(np.matmul(dif2[:,:,2],np.transpose(dif2[:,:,2])))/(A*L) # Erro Quadrático Médio plano B
    print("MSE_Red= {:.2e}".format(MSE_R), " MSE_Green= {:.2e}".format(MSE_G), " MSE_Blue= {:.2e}".format(MSE_B)+"- Imagem "+str(x))

    # # Cálculo da SNR de pico colorida (PSNR), 3 camadas R, G e B
    PSNR = 20*np.log10(255) - 10*np.log10(MSE_R + MSE_G + MSE_B)
    print("PSNR total = {:.2f} dB".format(PSNR))

    plt.figure(figsize=(20,20))
    infograf2 = "Imagem Reconstruída com PSNR total = " + str(np.uint8(PSNR)) + ' dB - Imagem '+str(x)
    plt.imshow(imgrec); plt.title(infograf2)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    MSE_Red= 7.85e-24  MSE_Green= 4.55e-24  MSE_Blue= 5.04e-24- Imagem 0
    PSNR total = 275.72 dB


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    MSE_Red= 6.32e-24  MSE_Green= 6.10e-24  MSE_Blue= 5.71e-24- Imagem 1
    PSNR total = 275.54 dB


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    MSE_Red= 6.45e-24  MSE_Green= 5.33e-24  MSE_Blue= 3.33e-24- Imagem 2
    PSNR total = 276.34 dB


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    MSE_Red= 4.17e-24  MSE_Green= 5.11e-24  MSE_Blue= 4.63e-24- Imagem 3
    PSNR total = 276.70 dB



    
![png](output_58_8.png)
    



    
![png](output_58_9.png)
    



    
![png](output_58_10.png)
    



    
![png](output_58_11.png)
    


# I. Repetindo etapas com foto montagem

### Com componente Y


```python
img = mpimg.imread('files/lab4/montagem.jpg')

img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

coefs2 = pywt.dwt2(img_gray,'haar', mode='periodization')  #1 nível de DWT
(cA, (cH, cV, cD)) = coefs2 #Separando os coeficientes
imgr = pywt.idwt2(coefs2, 'haar', mode = 'periodization')  #1 nível de IDWT

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(cA,'gray'); plt.title("CA - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV,'gray'); plt.title("CV - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH,'gray'); plt.title("CH - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD,'gray'); plt.title("CD - Bordas Diagonais")
```




    Text(0.5, 1.0, 'CD - Bordas Diagonais')




    
![png](output_61_1.png)
    



```python
# Calculo da MSE P&B
A, L, Camadas = img.shape
dif = img_gray - imgr
MSE_gray = np.sum(np.matmul(dif,np.transpose(dif)))/(A*L)
print("MSE_Y = {:.2e}".format(MSE_gray))
PSNR_Y = 20*np.log10(255) - 10*np.log10(MSE_gray)
print("PSNR_Luma = {:.2f} dB".format(PSNR_Y))
plt.figure(figsize=(20,10))
infograf = "Imagem Reconstruída de Luminância (Y) com PSNR = " + str(np.uint8(PSNR_Y)) + ' dB'
plt.subplot(1,2,1); plt.imshow(img_gray,'gray'); plt.title("Imagem Original P&B")
plt.subplot(1,2,2); plt.imshow(imgr,'gray'); plt.title(infograf)
```

    MSE_Y = 4.75e-24
    PSNR_Luma = 281.36 dB





    Text(0.5, 1.0, 'Imagem Reconstruída de Luminância (Y) com PSNR = 25 dB')




    
![png](output_62_2.png)
    



```python
C = pywt.wavedec2(img_gray,'haar', mode = 'symmetric', level=2) # Dois níveis de decomposição DWT
imgr2 = pywt.waverec2(C, 'haar', mode = 'symmetric') # Dois níveis de IDWT

# Para extrair os coeficientes de cada nível
cA2 = C[0]  # Coeficientes de Aproximação nível 2
(cH1, cV1, cD1) = C[-1] # Coeficientes de Detalhes nível 1
(cH2, cV2, cD2) = C[-2] # Coeficientes de Detalhes nível 2

# Imagem Original
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Plot dos coeficientes do nível 2
plt.figure(figsize=(5,5))
plt.subplot(2,2,1)
plt.imshow(cA2, 'gray'); plt.title('Ap. N2: CA2')
plt.subplot(2,2,2)
plt.imshow(cV2, 'gray'); plt.title('B. V. N2: CV2')
plt.subplot(2,2,3)
plt.imshow(cH2, 'gray'); plt.title('B. H. N2: CH2')
plt.subplot(2,2,4)
plt.imshow(cD2, 'gray'); plt.title('B. D. N2: CD2')

# Plot Original e Reconstrução
plt.figure(figsize=(20,10))
plt.subplot(1,2,1); plt.imshow(img_gray,'gray'); plt.title('Imagem Original')
plt.subplot(1,2,2); plt.imshow(imgr2,'gray'); plt.title('Imagem Reconstruída')
```




    Text(0.5, 1.0, 'Imagem Reconstruída')




    
![png](output_63_1.png)
    



    
![png](output_63_2.png)
    



```python
CV1 = cV1.copy()
CH1 = cH1.copy()
CD1 = cD1.copy()

CA2 = cA2.copy()
CH2 = cH2.copy()
CV2 = cV2.copy()
CD2 = cD2.copy()
# Matriz Final Completa
CA1 = np.bmat([[CA2,CV2],[CH2,CD2]])
CC = np.bmat([[CA1,CV1],[CH1,CD1]])

plt.figure(figsize=(20,20))
plt.imshow(CC,'gray')
plt.title('Codificação de Imagem em multinível com função wavedec2()')
```




    Text(0.5, 1.0, 'Codificação de Imagem em multinível com função wavedec2()')




    
![png](output_64_1.png)
    



```python
# Imagem Original

plt.figure(figsize=(20,20))
plt.imshow(img); plt.title("Imagem Original")

# Codificação por planos de cores
# Plano Vermelho
coefs_R = pywt.dwt2(img[:,:,0],'haar', mode='periodization')  #1 nível de DWT R
(cA_R, (cH_R, cV_R, cD_R)) = coefs_R #Separando os coeficientes
cr_R = pywt.idwt2(coefs_R, 'haar', mode = 'periodization')  #1 nível de IDWT R

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_R,'Reds_r'); plt.title("CA_Red - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_R,'Reds_r'); plt.title("CV_Red - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_R,'Reds_r'); plt.title("CH_Red - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_R,'Reds_r'); plt.title("CD_Red - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_R, 'Reds_r'); plt.title("Imagem Reconstruída Red")
```




    Text(0.5, 1.0, 'Imagem Reconstruída Red')




    
![png](output_65_1.png)
    



    
![png](output_65_2.png)
    



    
![png](output_65_3.png)
    



```python
# Plano Verde
coefs_G = pywt.dwt2(img[:,:,1],'haar', mode='periodization')  #1 nível de DWT G
(cA_G, (cH_G, cV_G, cD_G)) = coefs_G #Separando os coeficientes
cr_G = pywt.idwt2(coefs_G, 'haar', mode = 'periodization')  #1 nível de IDWT G

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_G,'Greens_r'); plt.title("CA_Green - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_G,'Greens_r'); plt.title("CV_Green - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_G,'Greens_r'); plt.title("CH_Green - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_G,'Greens_r'); plt.title("CD_Green - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_G, 'Greens_r'); plt.title("Imagem Reconstruída Green")
```




    Text(0.5, 1.0, 'Imagem Reconstruída Green')




    
![png](output_66_1.png)
    



    
![png](output_66_2.png)
    



```python
# Plano Azul
coefs_B = pywt.dwt2(img[:,:,2],'haar', mode='periodization')  #1 nível de DWT B
(cA_B, (cH_B, cV_B, cD_B)) = coefs_B #Separando os coeficientes
cr_B = pywt.idwt2(coefs_B, 'haar', mode = 'periodization')  #1 nível de IDWT B

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_B,'Blues_r'); plt.title("CA_Blue - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_B,'Blues_r'); plt.title("CV_Blue - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_B,'Blues_r'); plt.title("CH_Blue - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_B,'Blues_r'); plt.title("CD_Blue - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_B, 'Blues_r'); plt.title("Imagem Reconstruída Blue")
```




    Text(0.5, 1.0, 'Imagem Reconstruída Blue')




    
![png](output_67_1.png)
    



    
![png](output_67_2.png)
    



```python
# Reconstrução Nível 1 Colorida 
A1, L1 = cA_R.shape
imgrec1 = np.zeros((A1,L1,3))
imgrec1[:,:,0] = cA_R.copy()
imgrec1[:,:,1] = cA_G.copy()
imgrec1[:,:,2] = cA_B.copy()
plt.figure(figsize=(10,10))
plt.imshow(imgrec1); plt.title("Imagem Reconstruída DWT/IDWT Nível 1")
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).





    Text(0.5, 1.0, 'Imagem Reconstruída DWT/IDWT Nível 1')




    
![png](output_68_2.png)
    



```python
C1 = pywt.wavedec2(img_gray,'haar', mode = 'symmetric', level=1) # Um nível de decomposição DWT
CA1_ = 255 * C1[0] / np.abs(C1[0]).max()
CA1_ = CA1_.astype(int)

# aproximação de NÍVEL 2 - ja´obtido no item (D) e converte para inteiro
CA2_ = 255 * cA2 / np.abs(cA2).max()
CA2_ = CA2_.astype(int)

# Salva no drive
cv.imwrite('files/lab4/montagem_DWT_N1_Y.bmp', CA1_) # Aproximação Nível 1 só Y
cv.imwrite('files/lab4/montagem_DWT_N2_Y.bmp', CA2_) # Aproximação Nível 2 só Y 
```




    True




```python
# Reconstrução Colorida Original
A,L = imgr2.shape
imgrec = np.zeros((A,L,3))
imgrec[:,:,0] = cr_R.copy()
imgrec[:,:,1] = cr_G.copy()
imgrec[:,:,2] = cr_B.copy()

# Calculo do MSE colorida
dif2 = img - imgrec
MSE_R = np.sum(np.matmul(dif2[:,:,0],np.transpose(dif2[:,:,0])))/(A*L) # Erro Quadrático Médio plano R
MSE_G = np.sum(np.matmul(dif2[:,:,1],np.transpose(dif2[:,:,1])))/(A*L) # Erro Quadrático Médio plano G
MSE_B = np.sum(np.matmul(dif2[:,:,2],np.transpose(dif2[:,:,2])))/(A*L) # Erro Quadrático Médio plano B
print("MSE_Red= {:.2e}".format(MSE_R), " MSE_Green= {:.2e}".format(MSE_G), " MSE_Blue= {:.2e}".format(MSE_B))

# Cálculo da SNR de pico colorida (PSNR), 3 camadas R, G e B
PSNR = 20*np.log10(255) - 10*np.log10(MSE_R + MSE_G + MSE_B)
print("PSNR total = {:.2f} dB".format(PSNR))

plt.figure(figsize=(20,20))
infograf2 = "Imagem Reconstruída com PSNR total = " + str(np.uint8(PSNR)) + ' dB'
plt.imshow(imgrec); plt.title(infograf2)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    MSE_Red= 5.42e-24  MSE_Green= 4.60e-24  MSE_Blue= 4.07e-24
    PSNR total = 276.64 dB





    Text(0.5, 1.0, 'Imagem Reconstruída com PSNR total = 20 dB')




    
![png](output_70_3.png)
    


### Com componetem Cr


```python
# Codificação da foto-montagem do grupo em um nível com DWT para a componente Cr
img = mpimg.imread('files/lab4/montagem.jpg')

ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
y, cr, cb = cv.split(ycrcb)

#img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

coefs2 = pywt.dwt2(cr,'haar', mode='periodization')  #1 nível de DWT
(cA, (cH, cV, cD)) = coefs2 #Separando os coeficientes
imgr = pywt.idwt2(coefs2, 'haar', mode = 'periodization')  #1 nível de IDWT

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(cA,'gray'); plt.title("CA - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV,'gray'); plt.title("CV - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH,'gray'); plt.title("CH - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD,'gray'); plt.title("CD - Bordas Diagonais")

# Calculo da MSE P&B
A, L, Camadas = img.shape
dif = cr - imgr
MSE_cr = np.sum(np.matmul(dif,np.transpose(dif)))/(A*L)
print("MSE_cr = {:.2e}".format(MSE_cr))
PSNR_cr = 20*np.log10(255) - 10*np.log10(MSE_cr)
print("PSNR_Luma = {:.2f} dB".format(PSNR_cr))
plt.figure(figsize=(20,10))
infograf = "Imagem Reconstruída de Luminância (cr) com PSNR = " + str(np.uint8(PSNR_cr)) + ' dB'
plt.subplot(1,2,1); plt.imshow(cr,'gray'); plt.title("Imagem Original P&B")
plt.subplot(1,2,2); plt.imshow(imgr,'gray'); plt.title(infograf)

C = pywt.wavedec2(cr,'haar', mode = 'symmetric', level=2) # Dois níveis de decomposição DWT
imgr2 = pywt.waverec2(C, 'haar', mode = 'symmetric') # Dois níveis de IDWT

# Para extrair os coeficientes de cada nível
cA2 = C[0]  # Coeficientes de Aproximação nível 2
(cH1, cV1, cD1) = C[-1] # Coeficientes de Detalhes nível 1
(cH2, cV2, cD2) = C[-2] # Coeficientes de Detalhes nível 2

# Imagem Original
cr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Plot dos coeficientes do nível 2
plt.figure(figsize=(5,5))
plt.subplot(2,2,1)
plt.imshow(cA2, 'gray'); plt.title('Ap. N2: CA2')
plt.subplot(2,2,2)
plt.imshow(cV2, 'gray'); plt.title('B. V. N2: CV2')
plt.subplot(2,2,3)
plt.imshow(cH2, 'gray'); plt.title('B. H. N2: CH2')
plt.subplot(2,2,4)
plt.imshow(cD2, 'gray'); plt.title('B. D. N2: CD2')

# Plot Original e Reconstrução
plt.figure(figsize=(20,10))
plt.subplot(1,2,1); plt.imshow(cr,'gray'); plt.title('Imagem Original')
plt.subplot(1,2,2); plt.imshow(imgr2,'gray'); plt.title('Imagem Reconstruída')

CV1 = cV1.copy()
CH1 = cH1.copy()
CD1 = cD1.copy()

CA2 = cA2.copy()
CH2 = cH2.copy()
CV2 = cV2.copy()
CD2 = cD2.copy()
# Matriz Final Completa
CA1 = np.bmat([[CA2,CV2],[CH2,CD2]])
CC = np.bmat([[CA1,CV1],[CH1,CD1]])

plt.figure(figsize=(20,20))
plt.imshow(CC,'gray')
plt.title('Codificação de Imagem em multinível com função wavedec2()')

# Imagem Original

plt.figure(figsize=(20,20))
plt.imshow(img); plt.title("Imagem Original")

# Codificação por planos de cores
# Plano Vermelho
coefs_R = pywt.dwt2(img[:,:,0],'haar', mode='periodization')  #1 nível de DWT R
(cA_R, (cH_R, cV_R, cD_R)) = coefs_R #Separando os coeficientes
cr_R = pywt.idwt2(coefs_R, 'haar', mode = 'periodization')  #1 nível de IDWT R

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_R,'Reds_r'); plt.title("CA_Red - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_R,'Reds_r'); plt.title("CV_Red - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_R,'Reds_r'); plt.title("CH_Red - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_R,'Reds_r'); plt.title("CD_Red - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_R, 'Reds_r'); plt.title("Imagem Reconstruída Red")

# Plano Verde
coefs_G = pywt.dwt2(img[:,:,1],'haar', mode='periodization')  #1 nível de DWT G
(cA_G, (cH_G, cV_G, cD_G)) = coefs_G #Separando os coeficientes
cr_G = pywt.idwt2(coefs_G, 'haar', mode = 'periodization')  #1 nível de IDWT G

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_G,'Greens_r'); plt.title("CA_Green - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_G,'Greens_r'); plt.title("CV_Green - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_G,'Greens_r'); plt.title("CH_Green - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_G,'Greens_r'); plt.title("CD_Green - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_G, 'Greens_r'); plt.title("Imagem Reconstruída Green")

# Plano Azul
coefs_B = pywt.dwt2(img[:,:,2],'haar', mode='periodization')  #1 nível de DWT B
(cA_B, (cH_B, cV_B, cD_B)) = coefs_B #Separando os coeficientes
cr_B = pywt.idwt2(coefs_B, 'haar', mode = 'periodization')  #1 nível de IDWT B

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_B,'Blues_r'); plt.title("CA_Blue - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_B,'Blues_r'); plt.title("CV_Blue - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_B,'Blues_r'); plt.title("CH_Blue - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_B,'Blues_r'); plt.title("CD_Blue - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_B, 'Blues_r'); plt.title("Imagem Reconstruída Blue")

# Reconstrução Nível 1 Colorida 
A1, L1 = cA_R.shape
imgrec1 = np.zeros((A1,L1,3))
imgrec1[:,:,0] = cA_R.copy()
imgrec1[:,:,1] = cA_G.copy()
imgrec1[:,:,2] = cA_B.copy()
plt.figure(figsize=(10,10))
plt.imshow(imgrec1); plt.title("Imagem Reconstruída DWT/IDWT Nível 1")

# obtenha aproximação de NÍVEL 1 e converte para inteiro
C1 = pywt.wavedec2(img_gray,'haar', mode = 'symmetric', level=1) # Um nível de decomposição DWT
CA1_ = 255 * C1[0] / np.abs(C1[0]).max()
CA1_ = CA1_.astype(int)

# aproximação de NÍVEL 2 - ja´obtido no item (D) e converte para inteiro
CA2_ = 255 * cA2 / np.abs(cA2).max()
CA2_ = CA2_.astype(int)

# Salva no drive
cv.imwrite('files/lab4/montagem_Cr_DWT_N1_cr.bmp', CA1_) # Aproximação Nível 1 só Y
cv.imwrite('files/lab4/montagem_Cr_DWT_N2_cr.bmp', CA2_) # Aproximação Nível 2 só Y

# Aproximação Nível 1 Colorida 
# converte RGB para BGR e converte para inteiro
imgrec1_ = np.zeros((A1,L1,3))
imgrec1_[:,:,0] = imgrec1[:,:,2] 
imgrec1_[:,:,1] = imgrec1[:,:,1] 
imgrec1_[:,:,2] = imgrec1[:,:,0] 
imgrec1_ = ( 255 * imgrec1_ / np.abs(imgrec1_).max() ).astype(int)

# Salva no drive
cv.imwrite('files/lab4//montagem_DWT_N1_colorida_cr.bmp', imgrec1_) # Gravando Aproximação Nível 1 Colorida


# Reconstrução Colorida Original
A,L = imgr2.shape
imgrec = np.zeros((A,L,3))
imgrec[:,:,0] = cr_R.copy()
imgrec[:,:,1] = cr_G.copy()
imgrec[:,:,2] = cr_B.copy()

# Calculo do MSE colorida
dif2 = img - imgrec
MSE_R = np.sum(np.matmul(dif2[:,:,0],np.transpose(dif2[:,:,0])))/(A*L) # Erro Quadrático Médio plano R
MSE_G = np.sum(np.matmul(dif2[:,:,1],np.transpose(dif2[:,:,1])))/(A*L) # Erro Quadrático Médio plano G
MSE_B = np.sum(np.matmul(dif2[:,:,2],np.transpose(dif2[:,:,2])))/(A*L) # Erro Quadrático Médio plano B
print("MSE_Red= {:.2e}".format(MSE_R), " MSE_Green= {:.2e}".format(MSE_G), " MSE_Blue= {:.2e}".format(MSE_B))

# Cálculo da SNR de pico colorida (PSNR), 3 camadas R, G e B
PSNR = 20*np.log10(255) - 10*np.log10(MSE_R + MSE_G + MSE_B)
print("PSNR total = {:.2f} dB".format(PSNR))

plt.figure(figsize=(20,20))
infograf2 = "Imagem Reconstruída com PSNR total = " + str(np.uint8(PSNR)) + ' dB'
plt.imshow(imgrec); plt.title(infograf2)
```

    MSE_cr = 2.70e-24
    PSNR_Luma = 283.81 dB


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    MSE_Red= 5.42e-24  MSE_Green= 4.60e-24  MSE_Blue= 4.07e-24
    PSNR total = 276.64 dB





    Text(0.5, 1.0, 'Imagem Reconstruída com PSNR total = 20 dB')




    
![png](output_72_4.png)
    



    
![png](output_72_5.png)
    



    
![png](output_72_6.png)
    



    
![png](output_72_7.png)
    



    
![png](output_72_8.png)
    



    
![png](output_72_9.png)
    



    
![png](output_72_10.png)
    



    
![png](output_72_11.png)
    



    
![png](output_72_12.png)
    



    
![png](output_72_13.png)
    



    
![png](output_72_14.png)
    



    
![png](output_72_15.png)
    



    
![png](output_72_16.png)
    



    
![png](output_72_17.png)
    


### Componente Cb


```python
# Codificação da foto-montagem do grupo em um nível com DWT para a componente Cb
img = mpimg.imread('files/lab4/montagem.jpg')

ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
y, cr, cb = cv.split(ycrcb)

coefs2 = pywt.dwt2(cb,'haar', mode='periodization')  #1 nível de DWT
(cA, (cH, cV, cD)) = coefs2 #Separando os coeficientes
imgr = pywt.idwt2(coefs2, 'haar', mode = 'periodization')  #1 nível de IDWT

plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(cA,'gray'); plt.title("CA - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV,'gray'); plt.title("CV - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH,'gray'); plt.title("CH - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD,'gray'); plt.title("CD - Bordas Diagonais")

# Calculo da MSE P&B
A, L, Camadas = img.shape
dif = cb - imgr
MSE_cb = np.sum(np.matmul(dif,np.transpose(dif)))/(A*L)
print("MSE_cb = {:.2e}".format(MSE_cb))
PSNR_cb = 20*np.log10(255) - 10*np.log10(MSE_cb)
print("PSNR_Luma = {:.2f} dB".format(PSNR_cb))
plt.figure(figsize=(20,10))
infograf = "Imagem Reconstruída de Luminância (cb) com PSNR = " + str(np.uint8(PSNR_cb)) + ' dB'
plt.subplot(1,2,1); plt.imshow(cb,'gray'); plt.title("Imagem Original P&B")
plt.subplot(1,2,2); plt.imshow(imgr,'gray'); plt.title(infograf)

C = pywt.wavedec2(cb,'haar', mode = 'symmetric', level=2) # Dois níveis de decomposição DWT
imgr2 = pywt.waverec2(C, 'haar', mode = 'symmetric') # Dois níveis de IDWT

# Para extrair os coeficientes de cada nível
cA2 = C[0]  # Coeficientes de Aproximação nível 2
(cH1, cV1, cD1) = C[-1] # Coeficientes de Detalhes nível 1
(cH2, cV2, cD2) = C[-2] # Coeficientes de Detalhes nível 2

# Imagem Original
cb = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Plot dos coeficientes do nível 2
plt.figure(figsize=(5,5))
plt.subplot(2,2,1)
plt.imshow(cA2, 'gray'); plt.title('Ap. N2: CA2')
plt.subplot(2,2,2)
plt.imshow(cV2, 'gray'); plt.title('B. V. N2: CV2')
plt.subplot(2,2,3)
plt.imshow(cH2, 'gray'); plt.title('B. H. N2: CH2')
plt.subplot(2,2,4)
plt.imshow(cD2, 'gray'); plt.title('B. D. N2: CD2')

# Plot Original e Reconstrução
plt.figure(figsize=(20,10))
plt.subplot(1,2,1); plt.imshow(cb,'gray'); plt.title('Imagem Original')
plt.subplot(1,2,2); plt.imshow(imgr2,'gray'); plt.title('Imagem Reconstruída')

CV1 = cV1.copy()
CH1 = cH1.copy()
CD1 = cD1.copy()

CA2 = cA2.copy()
CH2 = cH2.copy()
CV2 = cV2.copy()
CD2 = cD2.copy()
# Matriz Final Completa
CA1 = np.bmat([[CA2,CV2],[CH2,CD2]])
CC = np.bmat([[CA1,CV1],[CH1,CD1]])

plt.figure(figsize=(20,20))
plt.imshow(CC,'gray')
plt.title('Codificação de Imagem em multinível com função wavedec2()')

# Imagem Original

plt.figure(figsize=(20,20))
plt.imshow(img); plt.title("Imagem Original")

# Codificação por planos de cores
# Plano Vermelho
coefs_R = pywt.dwt2(img[:,:,0],'haar', mode='periodization')  #1 nível de DWT R
(cA_R, (cH_R, cV_R, cD_R)) = coefs_R #Separando os coeficientes
cr_R = pywt.idwt2(coefs_R, 'haar', mode = 'periodization')  #1 nível de IDWT R

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_R,'Reds_r'); plt.title("CA_Red - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_R,'Reds_r'); plt.title("CV_Red - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_R,'Reds_r'); plt.title("CH_Red - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_R,'Reds_r'); plt.title("CD_Red - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_R, 'Reds_r'); plt.title("Imagem Reconstruída Red")

# Plano Verde
coefs_G = pywt.dwt2(img[:,:,1],'haar', mode='periodization')  #1 nível de DWT G
(cA_G, (cH_G, cV_G, cD_G)) = coefs_G #Separando os coeficientes
cr_G = pywt.idwt2(coefs_G, 'haar', mode = 'periodization')  #1 nível de IDWT G

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_G,'Greens_r'); plt.title("CA_Green - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_G,'Greens_r'); plt.title("CV_Green - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_G,'Greens_r'); plt.title("CH_Green - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_G,'Greens_r'); plt.title("CD_Green - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_G, 'Greens_r'); plt.title("Imagem Reconstruída Green")

# Plano Azul
coefs_B = pywt.dwt2(img[:,:,2],'haar', mode='periodization')  #1 nível de DWT B
(cA_B, (cH_B, cV_B, cD_B)) = coefs_B #Separando os coeficientes
cr_B = pywt.idwt2(coefs_B, 'haar', mode = 'periodization')  #1 nível de IDWT B

plt.figure(figsize=(20,20))
plt.subplot(2,2,1)
plt.imshow(cA_B,'Blues_r'); plt.title("CA_Blue - Aproximação")
plt.subplot(2,2,2)
plt.imshow(cV_B,'Blues_r'); plt.title("CV_Blue - Bordas Verticais")
plt.subplot(2,2,3)
plt.imshow(cH_B,'Blues_r'); plt.title("CH_Blue - Bordas Horizontais")
plt.subplot(2,2,4)
plt.imshow(cD_B,'Blues_r'); plt.title("CD_Blue - Bordas Diagonais")

plt.figure(figsize=(20,20))
plt.imshow(cr_B, 'Blues_r'); plt.title("Imagem Reconstruída Blue")

# Reconstrução Nível 1 Colorida 
A1, L1 = cA_R.shape
imgrec1 = np.zeros((A1,L1,3))
imgrec1[:,:,0] = cA_R.copy()
imgrec1[:,:,1] = cA_G.copy()
imgrec1[:,:,2] = cA_B.copy()
plt.figure(figsize=(10,10))
plt.imshow(imgrec1); plt.title("Imagem Reconstruída DWT/IDWT Nível 1")

# obtenha aproximação de NÍVEL 1 e converte para inteiro
C1 = pywt.wavedec2(cb,'haar', mode = 'symmetric', level=1) # Um nível de decomposição DWT
CA1_ = 255 * C1[0] / np.abs(C1[0]).max()
CA1_ = CA1_.astype(int)

# aproximação de NÍVEL 2 - ja´obtido no item (D) e converte para inteiro
CA2_ = 255 * cA2 / np.abs(cA2).max()
CA2_ = CA2_.astype(int)

# Salva no drive
cv.imwrite('files/lab4/montagemGrupo_DWT_N1_cb.bmp', CA1_) # Aproximação Nível 1 só Y
cv.imwrite('files/lab4/montagemGrupo_DWT_N2_cb.bmp', CA2_) # Aproximação Nível 2 só Y

# Aproximação Nível 1 Colorida 
# converte RGB para BGR e converte para inteiro
imgrec1_ = np.zeros((A1,L1,3))
imgrec1_[:,:,0] = imgrec1[:,:,2] 
imgrec1_[:,:,1] = imgrec1[:,:,1] 
imgrec1_[:,:,2] = imgrec1[:,:,0] 
imgrec1_ = ( 255 * imgrec1_ / np.abs(imgrec1_).max() ).astype(int)

# Salva no drive
cv.imwrite('files/lab4/montagemGrupo_DWT_N1_colorida_cb.bmp', imgrec1_) 


# Reconstrução Colorida Original
A,L = imgr2.shape
imgrec = np.zeros((A,L,3))
imgrec[:,:,0] = cr_R.copy()
imgrec[:,:,1] = cr_G.copy()
imgrec[:,:,2] = cr_B.copy()

# Calculo do MSE colorida
dif2 = img - imgrec
MSE_R = np.sum(np.matmul(dif2[:,:,0],np.transpose(dif2[:,:,0])))/(A*L) # Erro Quadrático Médio plano R
MSE_G = np.sum(np.matmul(dif2[:,:,1],np.transpose(dif2[:,:,1])))/(A*L) # Erro Quadrático Médio plano G
MSE_B = np.sum(np.matmul(dif2[:,:,2],np.transpose(dif2[:,:,2])))/(A*L) # Erro Quadrático Médio plano B
print("MSE_Red= {:.2e}".format(MSE_R), " MSE_Green= {:.2e}".format(MSE_G), " MSE_Blue= {:.2e}".format(MSE_B))

# Cálculo da SNR de pico colorida (PSNR), 3 camadas R, G e B
PSNR = 20*np.log10(255) - 10*np.log10(MSE_R + MSE_G + MSE_B)
print("PSNR total = {:.2f} dB".format(PSNR))

plt.figure(figsize=(20,20))
infograf2 = "Imagem Reconstruída com PSNR total = " + str(np.uint8(PSNR)) + ' dB'
plt.imshow(imgrec); plt.title(infograf2)
```

    MSE_cb = 3.62e-24
    PSNR_Luma = 282.54 dB


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    MSE_Red= 5.42e-24  MSE_Green= 4.60e-24  MSE_Blue= 4.07e-24
    PSNR total = 276.64 dB





    Text(0.5, 1.0, 'Imagem Reconstruída com PSNR total = 20 dB')




    
![png](output_74_4.png)
    



    
![png](output_74_5.png)
    



    
![png](output_74_6.png)
    



    
![png](output_74_7.png)
    



    
![png](output_74_8.png)
    



    
![png](output_74_9.png)
    



    
![png](output_74_10.png)
    



    
![png](output_74_11.png)
    



    
![png](output_74_12.png)
    



    
![png](output_74_13.png)
    



    
![png](output_74_14.png)
    



    
![png](output_74_15.png)
    



    
![png](output_74_16.png)
    



    
![png](output_74_17.png)
    

