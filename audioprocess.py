import numpy as np
import scipy.io.wavfile as wavfile
from ctypes import *
import pandas as pd
from  glob import glob
import os
import librosa
import python_speech_features as psf
from tqdm import tqdm


label=pd.read_csv('/data/xtx/yanghan/thirdtool/data/AIWIN/train/arkit.csv')
print('lable:',label)
all_label=[]
for per_col in label.columns:
    all_label.append(per_col)


def audio_mfcc(sig,rate,feature_file=None):

    videorate=25
    winlen=1./videorate
    winstep=0.5/videorate
    numcep=13
    winfunc=np.hanning

    mfcc=psf.mfcc(sig,rate,winlen=winlen,winstep=winstep,numcep=numcep,nfilt=numcep*2,nfft=int(rate/videorate),winfunc=winfunc)
    #print('------------mfcc.shape',mfcc.shape)
    mfcc_delta=psf.base.delta(mfcc,2)
    mfcc_delta2=psf.base.delta(mfcc_delta,2)
    mfcc_all=np.concatenate((mfcc,mfcc_delta,mfcc_delta2),axis=1)
    #print('mfcc_all.shape',mfcc_all.shape)

    if feature_file:
        np.save(feature_file, mfcc_all)


def audioTestProcess(path,outdata_dir):
    if not os.path.exists(outdata_dir):
        os.mkdir(outdata_dir)
    all_wavs=glob(path+"/*.wav")
    #all_csvs=glob(path+'/*.csv')
    #print('len(all_wavs)',len(all_wavs))
    #print('len(all_csvs)',len(all_csvs))
    #exit()

    for per_wav_path in tqdm(all_wavs):
        #print('----------------per_wav_path',per_wav_path)
        #rate,sig=wavfile.read(per_wav_path)

        feature_basename=per_wav_path.split('/')[-1].split('.')[0]
        #print('----------------feature_basename',feature_basename)
        sig,rate=librosa.load(per_wav_path,sr=48000)

        #print('rate',rate)
        #print('sig',sig.shape)
        frame_per_second=25
        chunks_lenght=260
        audio_framenum=int(len(sig)/rate*frame_per_second)

        a=np.zeros(chunks_lenght*rate//1000,dtype=np.int16)
        #print('a.shape',a.shape)
        signal=np.hstack((a,sig,a))
        #print('signal',signal.shape)

        frames_step=1000.0/frame_per_second
        rate_HKZ=int(rate/1000)


        audio_frames=[signal[int(i*frames_step*rate_HKZ):int((i*frames_step+chunks_lenght*2)*rate_HKZ)] for i in range(audio_framenum)]
        #print('len(audio_frame)',len(audio_frames))
        #assert len(audio_frames)==audio_blenshapnum
        for i in range(len(audio_frames)):
            #print(audio_frames[i].shape)
            audio_mfcc(audio_frames[i],rate=rate,feature_file=os.path.join(outdata_dir,feature_basename+'_{}_mfcc.npy'.format(str(i))))


    pass


def audioProcess(path,outdata_dir):
    if not os.path.exists(outdata_dir):
        os.mkdir(outdata_dir)
    all_wavs=glob(path+"/*.wav")
    all_csvs=glob(path+'/*.csv')
    #print('len(all_wavs)',len(all_wavs))
    #print('len(all_csvs)',len(all_csvs))
    #exit()

    for per_wav_path in tqdm(all_wavs):
        #print('----------------per_wav_path',per_wav_path)
        #rate,sig=wavfile.read(per_wav_path)

        feature_basename=per_wav_path.split('/')[-1].split('.')[0]
        #print('----------------feature_basename',feature_basename)
        sig,rate=librosa.load(per_wav_path,sr=48000)

        #print('rate',rate)
        #print('sig',sig.shape)
        frame_per_second=25
        chunks_lenght=260
        audio_framenum=int(len(sig)/rate*frame_per_second)
        #print('audio_framenum',audio_framenum)
        if per_wav_path.split('/')[-1].split('.')[0][0].isupper():
            per_csv_path=per_wav_path.split('.')[0]+'_anim.csv'
        else:
            per_csv_path=per_wav_path.split('.')[0]+'_Anim.csv'
        # print('--------------------per_csv_path',per_csv_path)
        # continue
        blendshape_num=os.path.join(path,per_csv_path)
        blendtensor=np.array(pd.read_csv(per_csv_path,usecols=all_label))
        #print('blendtensor.shape',blendtensor.shape)

        audio_blenshapnum=blendtensor.shape[0]

        #print('audio_blenshapnum',audio_blenshapnum)
        #assert audio_framenum==audio_blenshapnum



        a=np.zeros(chunks_lenght*rate//1000,dtype=np.int16)
        #print('a.shape',a.shape)
        signal=np.hstack((a,sig,a))
        #print('signal',signal.shape)

        frames_step=1000.0/frame_per_second
        rate_HKZ=int(rate/1000)


        audio_frames=[signal[int(i*frames_step*rate_HKZ):int((i*frames_step+chunks_lenght*2)*rate_HKZ)] for i in range(audio_blenshapnum)]
        #print('len(audio_frame)',len(audio_frames))
        #assert len(audio_frames)==audio_blenshapnum
        for i in range(len(audio_frames)):
            #print(audio_frames[i].shape)
            audio_mfcc(audio_frames[i],rate=rate,feature_file=os.path.join(outdata_dir,feature_basename+'_{}_mfcc.npy'.format(str(i))))







if __name__ == '__main__':
    #audioProcess(path='/data/xtx/yanghan/thirdtool/Audio2BS/data/train_val',outdata_dir='/data/xtx/yanghan/thirdtool/Audio2BS/data/train_val_feature_blendshape')
    audioTestProcess(path='/data/xtx/yanghan/thirdtool/Audio2BS/data/test-B/audio_for_B/all_test_B',outdata_dir='/data/xtx/yanghan/thirdtool/Audio2BS/data/test-B/audio_for_B/all_test_B_output')