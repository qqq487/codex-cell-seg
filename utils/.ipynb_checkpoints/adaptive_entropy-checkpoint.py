import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance

class DEE():
    def contrast_enhancement(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        gradient =  cv2.Laplacian(s,cv2.CV_32F, ksize = 1)  #cv2.Laplacian(s, cv2.CV_32F, ksize = 1) 
        clipped_gradient = gradient * np.exp(-1 * np.abs(gradient) * np.abs(s - 0.5))

        #normalize to [-1...1]
        clipped_gradient =  2*(clipped_gradient - np.max(clipped_gradient))/-np.ptp(clipped_gradient)-1
        clipped_gradient =  0.2 * clipped_gradient #--> 0.5 limits maximum saturation change to 50 %
        factor = np.add(1.0, clipped_gradient)

        s = np.multiply(s, factor)
        s = cv2.convertScaleAbs(s)

        v = self.adaptiveGammaCorrection(v)
        s = self.adaptiveCLAHE(s)
        
        final_CLAHE = cv2.merge((h,s,v))

        #additional sharpening
        tmpimg = cv2.cvtColor(final_CLAHE, cv2.COLOR_HSV2BGR)
        shimg = Image.fromarray(tmpimg)
        # sharpener = ImageEnhance.Sharpness(shimg)
        # result = sharpener.enhance(2.0)
        result = shimg
        
        return np.array(result)

    def adaptiveGammaCorrection(self, v_Channel):
        #calculate general variables
        I_in = v_Channel/255.0
        I_out = I_in
        
        sigma = np.std(I_in)
        mean = np.mean(I_in)
        D = 4*sigma

        #low contrast image
        if D <= 1/3:

            gamma = - np.log2(sigma)
            I_in_f = I_in**gamma
            mean_f = (mean**gamma)
            k =  I_in_f + (1 - I_in_f) * mean_f
            c = 1 / (1 + self.heaviside(0.5 - mean) * (k-1))
            #dark
            if mean < 0.5:
                I_out = I_in_f / ((I_in_f + ((1-I_in_f) * mean_f)))
            #bright			
            else:
                I_out = c * I_in_f
        #high contrast image
        elif D > 1/3:
            gamma = np.exp((1- (mean+sigma))/2)
            I_in_f = I_in**gamma
            mean_f = (mean**gamma)

            k =  I_in_f + (1 - I_in_f) * mean_f
            c = 1/ (1 + self.heaviside(0.5 - mean) * (k-1))
            I_out = c * I_in_f
        else:
            print('Error calculating D')
        I_out = I_out*255
        return I_out.astype(np.uint8)		

    def adaptiveCLAHE(self, channel):    
        channel_orig = channel
    
        #resizing for drastic speedup with minor quality loss
        channel = cv2.resize(channel, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) 

        basic_limit = 0.5
        expanding = 0.01

        res_entropys = []
        cur_entropy = self.calcEntropy(channel)
        res_entropys.append(cur_entropy)


        for cnt in range(50):
            tmp_v_CLAHE = self.CLAHE(channel, basic_limit + expanding*cnt, 8)
            cur_entropy = self.calcEntropy(tmp_v_CLAHE)
            res_entropys.append(cur_entropy)

        
        #find and apply optimal cliplimit
        res_entropys = list(map(float, res_entropys))
        opt_Limit = basic_limit  + expanding*self.calcCurvature(range(51), res_entropys)

        if opt_Limit < basic_limit:
            opt_Limit=basic_limit


        #adjust window size
        tiles = 6
        res_entropys = []
        tmp_v_CLAHE = self.CLAHE(channel, opt_Limit, 8)
        cur_entropy = self.calcEntropy(tmp_v_CLAHE)
        res_entropys.append(cur_entropy)

        for cnt in range(7):
            tmp_v_CLAHE = self.CLAHE(channel, opt_Limit, tiles + cnt)
            cur_entropy = self.calcEntropy(tmp_v_CLAHE)
            res_entropys.append(cur_entropy)


        res_entropys = list(map(float, res_entropys))
        opt_tiles = tiles + self.calcCurvature(range(8), res_entropys)

        #return optimized channel
        return self.CLAHE(channel_orig, opt_Limit, opt_tiles)

    @staticmethod
    def heaviside(x):
        if x <= 0:
            return 0
        else:
            return 1

    @staticmethod
    def CLAHE(channel, limit, tiles):
        clahe = cv2.createCLAHE(clipLimit= limit, tileGridSize=(tiles,tiles)) 
        return clahe.apply(channel)

    @staticmethod
    def calcEntropy(channel):
        hist = cv2.calcHist([channel],[0],None,[256],[0,256]) / channel.size
        entropy = np.sum(hist* np.log2(hist + 1e-7))
        return (-1.0 * entropy)	
    
    @staticmethod
    def calcCurvature(xs, ys):
        dx_dt = np.gradient(xs)
        dy_dt = np.gradient(ys)
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5

        #return optimal position
        return np.argmax(curvature)