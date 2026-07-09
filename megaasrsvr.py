#8ka
# conda activate /data/conda/envs/mega-asr
# cd /data/mega-asr/Mega-ASR
# python megaasrsvr.py
#yhy2026-03-18 没识别结果或者结果不是汉字的保存下来看看为啥
#yhy2026-03-20 SAVENOICE=1 识别为空也保存，=0识别为空不保存。
#yhy2026-03-24 gzip上传
#yhy2026-03-25 支持mp3上传 支持mp3 和aac 编码 url传递参数 voicecodec=mp3 启用
#yhy2026-03-28 支持多个节点NODEID区分
#yhy2026-05-29 支持返回语种。
#yhy2026-06-26 mysql获取热词写内存。使用hotword目录支持英文热词。pip install PyMySQL rapidfuzz -i https://pypi.doubanio.com/simple 
#/data/conda/envs/mega-asr/lib/python3.10/site-packages/qwen_asr
#yhy2026-07-08 ver1.0
import os
import sys
import signal
import logging
from logging.handlers import TimedRotatingFileHandler
from logging.handlers import RotatingFileHandler
import re
import time
import datetime
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.gen
import tornado.websocket
from tornado.concurrent import run_on_executor
from tornado.options import define, options
from concurrent.futures import ThreadPoolExecutor
from typing import List
import threading
import numpy as np
import argparse
from pathlib import Path
import torch
import torchaudio
import base64
import io
import urllib.request
from typing import Tuple
import json
import audioread
#import pypinyin
#from pypinyin import Style
#from hotword import PhonemeCorrector
#import pymysql.cursors
sys.path.append("src")
from MegaASR.model.megaASR import MegaASR
    
import struct
try:
    import codecs
except ImportError:
    codecs = None 
#模糊化    
mohu=[
    ['zh','z'],
    ['ch','c'],
    ['sh','s'],
    ['l','r'],
    ['ang','an'],
    ['eng','en'],
    ['ing','in'],
    ['ong','on'],
    ['iang','ian'],
    ['uang',' uan'],
    ['iong','ion'],
    ['ueng','uen'],
    ]
def checkhotword(userid,text,hotwords,lock):
    #lock.acquire()
    if userid not in hotwords:     
        #lock.release()    
        return text
    #t0 = time.perf_counter()
    result = hotwords[userid].correct(text)
    #lock.release()
    #t_ms = (time.perf_counter() - t0) * 1000
    #print(f"原始: {text}=>{result.text},耗时{t_ms:.1f}ms")
    log.info("userid=%s,%s=>%s",userid,text,result.text)
    return result.text
    
def addhotword(userid,hot_content,hotwords,lock):
    # 初始化
    pc = PhonemeCorrector(threshold=0.7)#0.85
    #print(pc)
    n_hw = pc.update_hotwords(hot_content)
    #print(f"userid {userid},热词 {n_hw} 个")
    log.error("userid=%s,hotword:%d个",userid,n_hw)
    #lock.acquire()      
    hotwords[userid]=pc
    #lock.release()  
    
def gethotbymysql(entid,hotwords,lock):
    try:
        conn = pymysql.connect(host=dbserver,port=dbport,
                                 user=dbuser,
                                 password=dbpass,
                                 database=dbdatabase,
                                 cursorclass=pymysql.cursors.DictCursor)

    except Exception as e:
        log.error("mysql connect出错:%s", e)
        return 
    cursor = conn.cursor()    
    try:
        if entid=="":
            sql = "SELECT `entid`,`hot` FROM `qu_hotinfo`"
            cursor.execute(sql)
            results = cursor.fetchall()
            for res in results:
                #print(row)
                userid=res["entid"] 
                hot=res["hot"]
                if len(userid)>1 and len(hot)>5:#至少2个汉字
                    addhotword(userid,hot,hotwords,lock)
        else:
            sql = "SELECT `hot` FROM `qu_hotinfo` WHERE `entid`=%s"
            cursor.execute(sql, (entid,))
            result = cursor.fetchone()
            if result is not None:
                hot=result["hot"] #dict
                if len(hot)>5:#至少2个汉字
                    addhotword(entid,hot,hotwords,lock) 
            else:
                log.error("mysql entid:%s,没查到", entid)         
    except Exception as e:
        log.error("mysql查询出错:%s", e)
    finally:
        # 关闭游标和连接
        cursor.close()
        conn.close()     
        
class MultiprocessHandler(logging.FileHandler):
    """支持多进程的TimedRotatingFileHandler"""
    def __init__(self,filename,when='H',backupCount=240,encoding=None,delay=False,maxBytes=2048000,):
        """filename 日志文件名,when 时间间隔的单位,backupCount 保留文件个数
        delay 是否开启 OutSteam缓存
            True 表示开启缓存，OutStream输出到缓存，待缓存区满后，刷新缓存区，并输出缓存数据到文件。
            False表示不缓存，OutStrea直接输出到文件"""
        self.prefix = filename
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.when = when.upper()
        self.extMath = r"^\d{4}-\d{2}-\d{2}"
        self.when_dict = {
            'S':"%Y-%m-%d-%H-%M-%S",
            'M':"%Y-%m-%d-%H-%M",
            'H':"%Y-%m-%d-%H",
            'D':"%Y-%m-%d"
        }
        #日志文件日期后缀
        self.suffix = self.when_dict.get(when)
        if not self.suffix:
            raise ValueError(u"指定的日期间隔单位无效: %s" % self.when)
        #拼接文件路径 格式化字符串
        self.filefmt = os.path.join("log","%s.%s.log" % (self.prefix,self.suffix))
        #使用当前时间，格式化文件格式化字符串
        self.filePath = datetime.datetime.now().strftime(self.filefmt)
        #获得文件夹路径
        _dir = os.path.dirname(self.filefmt)
        try:
            #如果日志文件夹不存在，则创建文件夹
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except Exception:
            #print "创建文件夹失败"
            #print "文件夹路径：" + self.filePath
            pass
        if codecs is None:
            encoding = None
        logging.FileHandler.__init__(self,self.filePath,'a+',encoding,delay)

    def shouldChangeFileToWrite(self):
        """更改日志写入目的写入文件
        :return True 表示已更改，False 表示未更改"""
        #以当前时间获得新日志文件路径
        _filePath = datetime.datetime.now().strftime(self.filefmt)
        if _filePath != self.filePath:
            self.filePath = _filePath
            return True
        return False

    def doChangeFile(self):
        """输出信息到日志文件，并删除多于保留个数的所有日志文件"""
        #日志文件的绝对路径
        self.baseFilename = os.path.abspath(self.filePath)
        if self.stream:
            self.stream.close()
            self.stream = None
        if not self.delay:
            self.stream = self._open()
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)

    def getFilesToDelete(self):
        """获得过期需要删除的日志文件"""
        # _ 表示占位符，没什么实际意义，
        dirName,_ = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        prefix = self.prefix + '.'
        plen = len(prefix)
        for fileName in fileNames:
            if fileName[:plen] == prefix:
                #日期后缀 mylog.2017-03-19 中的 2017-03-19
                suffix = fileName[plen:]
                if re.compile(self.extMath).match(suffix):
                    result.append(os.path.join(dirName,fileName))
        result.sort()
        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result

    def emit(self, record):
        """发送一个日志记录
        覆盖FileHandler中的emit方法，logging会自动调用此方法"""
        try:
            if self.shouldChangeFileToWrite():
                self.doChangeFile()
            logging.FileHandler.emit(self,record)
        except (KeyboardInterrupt,SystemExit):
            raise
        except:
            self.handleError(record)
def readfile2list(test_file: str)->list:    
    result=[]
    fp=open(test_file,"r", encoding='utf-8') 
    lines = fp.read().split('\n')
    for l in lines:        
        l=l.replace("\r","")
        if len(l)>0:
            result.append(l)
    fp.close()
    return     result            
def GetConfigParameter(cmd,default,configfile):
    result=default    
    if not os.path.exists(configfile):
        return result
    params=readfile2list(configfile)
    for ls in params:        
        param=ls.split('=')
        if cmd==param[0]:
            result=param[1]
            break
    return result  
def GetHotWord(configfile):
    result=[]    #[userid,[[hz],[py]]]
    if not os.path.exists(configfile):
        return result
    lines=readfile2list(configfile)
    for line in lines:        
        if len(line)<2 or line[0]=='#':#最小两个字
            continue
        param=line.split(' ')
        lens=len(param)
        if lens<1:#非法
            continue
        if lens==1:#只有汉字
            userid="DPS_SYS"
            hz=param[0]
        else:
            userid=param[0]
            hz=param[1]
        if IsContainChinese(hz)==False:
            log.error('hot hz=%s,skip',hz)
            continue
        
        pylist=[]
        hzlist = [char for char in hz]
        if lens<=2:#TEST 陈良                  
            pylist1 = pypinyin.pinyin(hzlist, style=pypinyin.NORMAL)            
            for r in pylist1:
                s=r[0]
                for m in mohu:
                    s=s.replace(m[0],m[1])
                pylist.append(s)
        else:#DPS_SYS 陈益森 chen yi sen
            pylist1=param[2:]
            for r in pylist1:
                s=r
                for m in mohu:
                    s=s.replace(m[0],m[1])
                pylist.append(s)
        #print(pylist) 
        lens=len(result)
        find=0
        for i in range(lens):        
            if result[i][0]==userid:
                result[i][1].append((hzlist,pylist))   
                #print(result)
                find=1
                break
        if find==0:
            result.append((userid,[(hzlist,pylist)]))
            #print(result)
    #print(result)
    return result
def sublist_index(main_list, sub_list):
    for i in range(len(main_list)-len(sub_list)+1):
        if all(main_list[i+j] == sub_list[j] for j in range(len(sub_list))):
            return i
    return -1
def language2lan(language):
    if language=="":
        return ""  
    if language=="Chinese":  
        return "zh"
    if language=="English":  
        return "en"
    if language=="Japanese": 
        return "ja"
    if language=="Cantonese":
        return "yue"
    if language=="Russian": 
        return "ru"
    if language=="Arabic": 
        return "ar"
    if language=="Spanish": 
        return "es"
    if language=="Italian": 
        return "it"
    if language=="Vietnamese": 
        return "vi"
    if language=="French": 
        return "fr"
    if language=="German": 
        return "de"
    if language=="Portuguese": 
        return "pt"
    if language=="Turkish": 
        return "tr"
    if language=="Polish": 
        return "pl"
    if language=="Dutch": 
        return "nl"
    if language=="Swedish": 
        return "sv"
    if language=="Indonesian": 
        return "id"
    if language=="Hindi": 
        return "hi"
    if language=="Finnish": 
        return "fi"
    if language=="Greek": 
        return "el"
    if language=="Malay": 
        return "ms"
    if language=="Czech": 
        return "cs"
    if language=="Romanian": 
        return "ro"
    if language=="Danish": 
        return "da"
    if language=="Hungarian": 
        return "hu"
    if language=="Persian": 
        return "fa" #波斯语
    if language=="Macedonian": 
        return "mk"
    if language=="Filipino": 
        return "fil"        
    #cython Structural pattern match is not yet implemente
    '''match language:
        case "Chinese":  
            return "zh"
        case "English":  
            return "en"
        case "Japanese": 
            return "ja"
        case "Cantonese":
            return "yue"
        case "Russian": 
            return "ru"
        case "Arabic": 
            return "ar"
        case "Spanish": 
            return "es"
        case "Italian": 
            return "it"
        case "Vietnamese": 
            return "vi"
        case "French": 
            return "fr"
        case "German": 
            return "de"
        case "Portuguese": 
            return "pt"
        case "Turkish": 
            return "tr"
        case "Polish": 
            return "pl"
        case "Dutch": 
            return "nl"
        case "Swedish": 
            return "sv"
        case "Indonesian": 
            return "id"
        case "Hindi": 
            return "hi"
        case "Finnish": 
            return "fi"
        case "Greek": 
            return "el"
        case "Malay": 
            return "ms"
        case "Czech": 
            return "cs"
        case "Romanian": 
            return "ro"
        case "Danish": 
            return "da"
        case "Hungarian": 
            return "hu"
        case "Persian": 
            return "fa" #波斯语
        case "Macedonian": 
            return "mk"
        case "Filipino": 
            return "fil"
            '''
    return "zh" 
def CheckHotWord(userid,hztext,hotword):      
    hzlist = [char for char in hztext]
    py = pypinyin.pinyin(hzlist, style=pypinyin.NORMAL)
    pylist=[]
    for r in py:
        s=r[0]
        for m in mohu:
            s=s.replace(m[0],m[1])
        pylist.append(s)
    #print(pylist)               
    for userlist in hotword:
        if userid==userlist[0] or userlist[0]=="DPS_SYS":
            for hot in userlist[1]:
                pos=0
                tmp=pylist 
                while True:                    
                    index = sublist_index(tmp, hot[1])
                    if index != -1:
                        #print("hit in hotword", pos+index)
                        #print("hit in hotword", hot[0])
                        hzhot=hot[0]
                        c=0    
                        for r in hzhot:
                            hzlist[pos+index+c]=r
                            c=c+1
                        #print(hztext)
                        #print("".join(hzlist))
                        find=1
                        pos=pos+index+c
                        tmp=pylist[pos:]
                    else:
                        break                    
    return "".join(hzlist)     

torch.set_num_threads(1) #否则VAD cpu 100% 
tmp=GetConfigParameter("GPU","7","./asr.conf")
os.environ["CUDA_VISIBLE_DEVICES"] = tmp 
device=GetConfigParameter("device","cuda","./asr.conf")#device="cuda" #if torch.cuda.is_available() else "cpu")
log_fmt = "%(asctime)s %(message)s"
formatter = logging.Formatter(log_fmt)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
logfilename="log"+rq
tmp=GetConfigParameter("LOGTIME","H","./asr.conf")
log_file_handler = MultiprocessHandler(filename=logfilename,when=tmp,backupCount=30,delay=True)
log_file_handler.setFormatter(formatter)
log = logging.getLogger()
tmp=GetConfigParameter("LOGLEVEL","14","./asr.conf")
log_level=int(tmp)    
if log_level>10:
    log_level=log_level-10
    if log_level>=5:
        logging.basicConfig(level=logging.DEBUG)
    if log_level==4:
        logging.basicConfig(level=logging.INFO)
    if log_level==3:
        logging.basicConfig(level=logging.WARNING)    
    if log_level==2:
        logging.basicConfig(level=logging.ERROR)    
    if log_level<=1:
        logging.basicConfig(level=logging.FATAL)        
else:    
    if log_level>=5:
        log.setLevel(level=logging.DEBUG)
    if log_level==4:
        log.setLevel(level=logging.INFO)
    if log_level==3:
        log.setLevel(level=logging.WARNING)    
    if log_level==2:
        log.setLevel(level=logging.ERROR)    
    if log_level<=1:
        log.setLevel(level=logging.FATAL)                
log.addHandler(log_file_handler) 
loopcount=0    
maxuse=0
userid=0
token=0
tmp=GetConfigParameter("SAMPLES","16000","./asr.conf")
samples=int(tmp)
tmp=GetConfigParameter("MAX_VOICETIME","20","./asr.conf") #分段识别，一段15s
max_voice=int(tmp)*samples 
max_voice_ms=int(tmp)*1000 
tmp=GetConfigParameter("SAVEVOICE","0","./asr.conf")
savevoice=int(tmp)
tmp=GetConfigParameter("SAVENOICE","0","./asr.conf")
savenoice=int(tmp)
tmp=GetConfigParameter("EOS","1000","./asr.conf")
eos=int(tmp)/10
nodeid="mg"+GetConfigParameter("PORT","9375","./asr.conf")#yhy2026-03-28 支持多个节点NODEID区分
tmp=GetConfigParameter("CHECK_HOTWORD","0","./asr.conf")
check_hotword=int(tmp) 
voice_path=GetConfigParameter("VOICEPATH","./wav","./asr.conf")    
gcontext=GetConfigParameter("CONTEXT","","./asr.conf")#语音可能是多人嘈杂的噪声
if gcontext!="":
    gcontext2=gcontext+"。"
else:
    gcontext2=""    
tmp=GetConfigParameter("GPU_MEMORY","0.7","./asr.conf")
gpu_memory_utilization=float(tmp)    
#stream 要设置 max_new_tokens=32 max_inference_batch_size=-1 usevllm=1
tmp=GetConfigParameter("max_new_tokens","512","./asr.conf")
max_new_tokens=int(tmp)    
tmp=GetConfigParameter("max_inference_batch_size","1","./asr.conf")
max_inference_batch_size=int(tmp)  
tmp=GetConfigParameter("VLLM","0","./asr.conf")
usevllm=int(tmp)     
tmp=GetConfigParameter("STREAM","0","./asr.conf") 
stream=int(tmp) 
if stream>0:
    max_inference_batch_size=-1
    max_new_tokens=32
    usevllm=1
dbserver=GetConfigParameter("dbserver","127.0.0.1","./asr.conf")    
tmp=GetConfigParameter("dbport","3306","./asr.conf")    
dbport=int(tmp) 
dbuser=GetConfigParameter("dbuser","root","./asr.conf")    
dbpass=GetConfigParameter("dbpass","wdupec","./asr.conf")    
dbdatabase=GetConfigParameter("dbdatabase","test","./asr.conf")    

trans_url=GetConfigParameter("llamacpp_url","http://111.7.98.36:9398/v1/chat/completions","./asr.conf")    
def find_continuous_chinese_digits(text):
    chinese_digits = '幺一二三四五六七八九十'
    i = 0
    c=0
    while i < len(text):
        # 寻找连续的汉字数字子串的起始位置
        if text[i] in chinese_digits:
            start = i
            while i < len(text) and text[i] in chinese_digits:
                i += 1
                if i-start>=5:
                    #print("5")
                    return 5
            # 提取并输出连续的汉字数字子串
            #print(text[start:i])
            if i-start>c:
                c=i-start
                if c>=5:
                    #print(c)
                    return c
        else:
            i += 1
    #print(c)
    return c
    
def remove_consecutive_duplicates(text):
    # 使用正则表达式匹配连续重复3次的中文文本只有保留一次
    pattern = r'(.+?)\1{2,}'
    # 将连续重复的部分替换为单个实例
    result = re.sub(pattern, r'\1', text)
    return result
def Save2Wav(audio,filename):
    with open(filename, 'wb') as f:
        if audio[0]!=0x52 or audio[1]!=0x49 or audio[2]!=0x46 or audio[3]!=0x46:     
        # 构建 WAV 文件头
            audiolen=len(audio)
            header = struct.pack(
                '<4sI4s4sIHHIIHH4sI',
                b'RIFF', # ChunkID
                36 + audiolen, # ChunkSize
                b'WAVE', # Format
                b'fmt ', # Subchunk1ID
                16, # Subchunk1Size (PCM = 16)
                1, # AudioFormat (PCM = 1)
                1, # NumChannels
                16000, # SampleRate
                32000, # ByteRate
                2, # BlockAlign
                16, # BitsPerSample
                b'data', # Subchunk2ID
                audiolen # Subchunk2Size
            )
            # 写入头信息和 PCM 数据到输出文件
            f.write(header)
        f.write(audio)
    return filename
 

def IsContainChinese(text):
    pattern = re.compile(r'[\u4e00-\u9fff]')  # 中文字符的Unicode范围
    return bool(pattern.search(text))
#yhy2026-01-28增加 llama.cpp 接口使用 腾讯的HY-MT1.5-1.8B-Q4_K_M.gguf 翻译模型
def get_language(lan,flag):
    languages=[
      ["zh","中文","Chinese"], 
      ["en","英语","English"],
      ["ja","日语","Japanese"],
      ["ko","韩语","Korean"],
      ["yue","粤语","Cantonese"],
      ["ru","俄语","Russian"],
      ["ar","阿拉伯语","Arabic"],
      ["es","西班牙语","Spanish"],
      ["it","意大利语","Italian"],
      ["vi","越南语","Vietnamese"],
      ["fr","法语","French"],
      ["de","德语","German"],
      ["pt","葡萄牙语","Portuguese"],
      ["tr","土耳其语","Turkish"],
      ["pl","波兰语","Polish"],
      ["nl","荷兰语","Dutch"],  
      ["id","印尼语","Indonesian"],
      ["hi","印地语","Hindi"],
      ["he","希伯来语","Hebrew"],
      ["uk","乌克兰语","Ukrainian"],
      ["ms","马来语","Malay"],
      ["cs","捷克语","Czech"],
      ["ta","泰米尔语","Tamil"],
      ["th","泰语","Thai"],
      ["ur","乌尔都语","Urdu"],
      ["te","泰卢固语","Telugu"],
      ["fa","波斯语","Persian"],
      ["bn","孟加拉语","Bengali"],
      ["mn","蒙古语","Mongolian"],
      ["kk","哈萨克语","Kazakh"],
      ["mr","马拉地语","Marathi"],
      ["km","高棉语","Khmer"],
      ["gu","古吉拉特语","Gujarati"],
      ["my","缅甸语","Myanmar"],
      ["bo","藏语","Tibetan"],
      ["tl","菲律宾语","Tagalog"],
      ["ug","维吾尔语","Uyghur"]
    ]
    language=""
    for j in range(38):           
        if(lan==languages[j][0]):
            language=languages[j][flag];
            break
    return language;
'''def filter_str(sentence):
    sentence = re.sub(remove_nota, '', sentence)
    sentence = sentence.translate(remove_punctuation_map)
    return sentence.strip()    
# 判断中日韩英
def judge_language(s):
    s = filter_str(s)
    s = re.sub('[0-9]', '', s).strip()
    # unicode korean
    re_words = re.compile(u"[\uac00-\ud7ff]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    #res2 = re.sub(u"[\uac00-\ud7ff]+", '', s).strip()
    if len(res) > 1:
        return 'ko'
    # unicode japanese katakana and unicode japanese hiragana
    re_words = re.compile(u"[\u30a0-\u30ff\u3040-\u309f]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    #res2 = re.sub(u"[\u30a0-\u30ff\u3040-\u309f]+", '', s).strip()
    if len(res) > 1:
        return 'ja'
    # unicode chinese
    re_words = re.compile(u"[\u4e00-\u9fa5]+")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    #res2 = re.sub(u"[\u4e00-\u9fa5]+", '', s).strip()
    if len(res) > 1:
        return 'zh'   
    # unicode english
    re_words = re.compile(u"[a-zA-Z]")
    res = re.findall(re_words, s)  # 查询出所有的匹配字符串
    res2 = re.sub('[a-zA-Z]', '', s).strip()
    if len(res) >= 3:
        return 'en'        
    return 'unk'
'''        
def llamacpp_hz2en(text,from_lan,to_lan):
    fromlan=from_lan
    tolan=to_lan
    text1=text 
    result=text 
    lanflag="1"    
    if(fromlan==to_lan):#识别结果是目标语种
        return result,lanflag
    if(fromlan=="cn" or fromlan==""):
        fromlan="zh"
    if(tolan=="cn" or tolan==""): 
        tolan="zh"
    if(fromlan==tolan):
        return result,lanflag
    lanflag="0"   
    jsons=""
    #language_code=judge_language(text)
    #log.error("text=%s,language_code=%s,fromlan=%s,tolan=%s,judge_language",text,language_code,fromlan,tolan)      
    if(fromlan=="zh" or fromlan=="yue" or tolan=="zh" or tolan=="yue"):
        language=get_language(tolan,1)#中文语种名称
        jsons="{\"messages\": [{\"role\": \"system\",\"content\": \"You are a helpful assistant.\"},{\"role\": \"user\",\"content\": \"将以下文本翻译为"+language+"："+text1+"\"}]}"
    else:
        language=get_language(fromlan,2)
        jsons="{\"messages\": [{\"role\": \"system\",\"content\": \"You are a helpful assistant.\"},{\"role\": \"user\",\"content\": \"Translate the following segment into "+language+": "+text1+"\"}]}"
    # 发送POST请求
    data = jsons.encode()
    headers = {'Content-Type': 'application/json'}
    req = urllib.request.Request(url=trans_url,headers=headers, data=data, method='POST')
    response = urllib.request.urlopen(req)
    str=response.read().decode('utf-8')
    js=json.loads(str)
    #print(js)
    result=js['choices'][0]['message']['content']
    #{'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'role': 'assistant', 'content': 'I made a mistake just now. I actually said “three thousand”.'}}], 'created': 1771632770, 'model': 'HY-MT1.5-1.8B-Q4_K_M.gguf', 'system_fingerprint': 'b7913-a3fa03582', 'object': 'chat.completion', 'usage': {'completion_tokens': 15, 'prompt_tokens': 34, 'total_tokens': 49}, 'id': 'chatcmpl-C0Yh2X7S9BcYMEjQcPky4PD5xq0C72gE', 'timings': {'cache_n': 33, 'prompt_n': 1, 'prompt_ms': 70.394, 'prompt_per_token_ms': 70.394, 'prompt_per_second': 14.205756172401056, 'predicted_n': 15, 'predicted_ms': 99.834, 'predicted_per_token_ms': 6.655600000000001, 'predicted_per_second': 150.2494140272853}}
    print(result)
    return result,lanflag
     
class Index(tornado.web.RequestHandler):
    # 封装一个类
    def get(self):
        # get请求进入该方法
        # 返回字符串
        self.write('use:/dotcasr')
        
class DoTcAsr(tornado.web.RequestHandler):
    #起线程池，由当前RequestHandler持有
    executor = ThreadPoolExecutor(max_workers=128)
    def initialize(self, model,license,lock,hotword):
        self.model = model
        self.license=license
        self.lock=lock
        self.hotword=hotword
    def get(self):
        self.write('get not support.')
    @tornado.gen.coroutine
    def post(self):
        userid =self.get_argument('userid','') 
        token =self.get_argument('token','') 
        lanid =self.get_argument('lanid','') 
        fromlan =self.get_argument('fromlan','') 
        tolan =self.get_argument('tolan','') 
        voicecodec =self.get_argument('voicecodec','')
        entid =self.get_argument('entid','') 
        if entid!="":
            userid=entid
        if token!="updatehotword" and (userid=="" or token==""):
            if userid=="":
                result="{\"result\":\"\",\"errCode\":\"2012\",\"Msg\":\"userid=''!\"}"
            else:
                result="{\"result\":\"\",\"errCode\":\"2018\",\"Msg\":\"token=''!\"}"
            log.error("userid=%s,token=%s,result=%s",userid,token,result)  
            self.write(result)            
            return 
        else:
            if token=="updatehotword":#yhy2026-06-26 更新热词
                start1 = time.time()
                if check_hotword>0:
                    gethotbymysql(userid,self.hotword,self.lock)
                end1 = time.time()
                timems=(end1-start1)*1000
                result="{\"result\":\"\",\"errCode\":\"0\",\"Msg\":\"Update OK!\"}"
                log.error("userid=%s,time:%dms,result=%s",userid,timems,result)
                self.write(result)
                return
            file_metas = self.request.files.get('file', None)  
            if not file_metas:
                body=self.request.body 
                if not body:#
                    result="{\"result\":\"\",\"errCode\":\"-3\",\"Msg\":\"body=''!\"}" 
                    log.error("userid=%s,token=%s,result=%s",userid,token,result) 
                    self.write(result)            
                    return                     
            else:
                body=file_metas[0]['body']
            filelen=len(body)
            if filelen<320:
                result="{\"result\":\"\",\"errCode\":\"-3\",\"Msg\":\"file=''!\"}" 
                log.error("userid=%s,token=%s,result=%s",userid,token,result)            
            else:                
                if voicecodec=="" and filelen<16000:
                    result="{\"result\":\"\",\"errCode\":\"-1\",\"Msg\":\"file <500ms !\"}" 
                    log.error("userid=%s,token=%s,result=%s",userid,token,result)
                else:
                    global loopcount
                    loopcount=loopcount+1
                    count=loopcount
                    #log.info("userid=%s,count=%07d,ftime=%dms...",userid,count,filelen/32)
                    start1 = time.time()
                    result=yield self.worker_thread(userid,lanid,fromlan,tolan,count,body,voicecodec) #yhy2022-01-19 多进程只能使用文件 使用meta['body'])会卡住 
                    end1 = time.time()  
                    timems=(end1-start1)*1000
                    if voicecodec=="":
                        log.error("userid=%s,count=%07d,ftime=%dms,asrtime:%dms,rtf=%.4f,result=%s",userid,count,filelen/32,timems,timems*32/filelen,result)
                    else:
                        log.error("userid=%s,count=%07d,ftime=%dms,asrtime:%dms,rtf=%.4f,result=%s",userid,count,filelen/4,timems,timems*4/filelen,result)
        self.write(result)
    @run_on_executor
    def worker_thread(self,userid,lanid,fromlan,tolan,count,audio1,voicecodec):
        global maxuse    
        errCode="0"
        results="" 
        filename=""
        if lanid=="":
            language=None  
        elif lanid=="0":
            language="Chinese"#            Chinese (zh),  
        elif lanid=="1":
            language="English"#            English (en),
        elif lanid=="2":
            language="Japanese"#             
        elif lanid=="3":
            language="Korean"#             
        elif lanid=="4":
            language="Cantonese"#             
        elif lanid=="5": 
            language="Russian"#             
        elif lanid=="6": 
            language="Arabic"#             
        elif lanid=="7": 
            language="Spanish"#             
        elif lanid=="8":
            language="Italian"#             
        elif lanid=="9": 
            language="Vietnamese"#             
        elif lanid=="10": 
            language="French"#             
        elif lanid=="11": 
            language="German"#             
        elif lanid=="12": 
            language="Portuguese"#             
        elif lanid=="13": 
            language="Turkish"#
        elif lanid=="14": 
            language="Polish"#
        elif lanid=="16": 
            language="Dutch"#
        elif lanid=="17": 
            language="Swedish"#
        elif lanid=="18": 
            language="Indonesian"#
        elif lanid=="19": 
            language="Hindi"#
        elif lanid=="20": 
            language="Finnish"#
        elif lanid=="23": 
            language="Greek"#
        elif lanid=="24": 
            language="Malay"#
        elif lanid=="25": 
            language="Czech"#
        elif lanid=="26": 
            language="Romanian"#
        elif lanid=="27": 
            language="Danish"#
        elif lanid=="28": 
            language="Hungarian"#
        elif lanid=="42":
            language="Persian"#
        elif lanid=="50":
            language="Macedonian"#
        elif lanid=="90":
            language="Filipino"#tagalog  菲律宾语 (fil) 
        else:
            language=None 
        asrlanguage=""            
        start = time.time()   
        if voicecodec=="mp3":
            filename="/dev/shm/voc"+nodeid+"_"+str(count)+"_"+str(len(audio1))+".mp3"
            fmp3 = open(filename, 'wb')
            fmp3.write(audio1) 
            fmp3.close()
            with audioread.audio_open(filename) as f:
                c=0;
                for buf in f:                    
                    if c==0:
                        audio=buf
                    else:
                        audio=audio+buf
                    c=c+1
            os.remove(filename)
        else:
            audio=audio1        
        licuse=0
        if self.license>0:#控制授权
            while True:                
                self.lock.acquire()
                if maxuse<self.license:
                    maxuse=maxuse+1
                    licuse=maxuse
                self.lock.release()
                if licuse>0:
                    break
                time.sleep(0.02)#如果列表是空，循环等待20ms         
                continue    
            end = time.time() 
            wtime=(end - start)*1000
            if wtime>=200:
                log.error("userid=%s,count=%07d,lictime:%d ms,licuse=%d",userid,count,wtime,licuse)            
        try:
            if audio[0]==0x52 and audio[1]==0x49 and audio[2]==0x46 and audio[3]==0x46:
                pcm16=np.frombuffer(audio[44:],dtype=np.int16)
            else:
                pcm16=np.frombuffer(audio,dtype=np.int16)
            pcm16=pcm16.astype(float)*0.000030517578125
            audiolen=len(pcm16)
            if audiolen<max_voice:#//<20s 最大一次00s 16K单声道PCM语音
                #log.info("userid=%s,count=%07d,audiolen=%d,pos=0",userid,count,audiolen/samples)
                asrresult = self.model.infer(audio=(pcm16, samples),return_objects=True, language=language)
                text=""
                for i, r in enumerate(asrresult): 
                    text=text+r.text
                    asrlanguage=r.language                                           
                #print(results)     
                if len(text)>=30:#他可能只会他那个逻辑可能只是那种很嘈杂的声音但是因为这个抖音放的是阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰
                    text=text.replace("幺幺幺幺幺","")
                    digitcount=find_continuous_chinese_digits(text)
                    if digitcount<5:#95555
                        text = remove_consecutive_duplicates(text)
                results=text
            else:
                c=0
                pos=0
                while True:
                    lens=max_voice
                    if  pos+max_voice<audiolen and pos+max_voice+max_voice>audiolen:
                        lens=max_voice//2  #最后两段平均分配
                    if  pos+max_voice>audiolen:
                        lens=audiolen-pos
                    if lens<8000:
                         break
                    #log.info("userid=%s,count=%07d,audiolen=%d,pos=%d,len=%d",userid,count,audiolen/samples,pos/samples,lens/samples)
                    asrresult = self.model.infer(audio=(pcm16, samples),return_objects=True, language=language)
                    text=""
                    for i, r in enumerate(asrresult): 
                        text=text+r.text
                        asrlanguage=r.language                             
                    if len(text)>=30:#他可能只会他那个逻辑可能只是那种很嘈杂的声音但是因为这个抖音放的是阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰以前算过的阳光玫瑰
                        text=text.replace("幺幺幺幺幺","")
                        digitcount=find_continuous_chinese_digits(text)
                        if digitcount<5:#95555
                            text = remove_consecutive_duplicates(text)                    
                    results=results+text
                    #log.info("userid=%s,count=%07d,pos=%dms,result=%s",userid,count,pos/16,text)
                    pos=pos+lens                   
                    c=c+1
                    if pos+8000>=audiolen:
                        break           
        except Exception as e:
            Msg=str(e)
            log.error("userid=%s,count=%d,except=%s!",userid,count,Msg)
            errCode="-2"
            #result="{\"result\":\"\",\"errCode\":\"-2\",\"Msg\":\""+Msg+"\"}"
        if self.license>0:#控制授权
            self.lock.acquire()
            maxuse=maxuse-1
            licuse=maxuse
            self.lock.release()             
        if len(results)<=2:#//yhy2026-02-04垃圾去掉
            if savevoice==1 and results!="":
                folder_path=voice_path+"/"+time.strftime("%Y-%m-%d",time.localtime(start))
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                filename=folder_path+"/noise_"+userid+"_"+time.strftime("%Y-%m-%d-%H-%M-%S_",time.localtime(start))+str(count)+".wav"
                Save2Wav(audio,filename)
                log.error("userid=%s,count=%07d,filename=%s,to hz!,text=[%s]",userid,count,filename,results)
            results=""            
        else:
            if gcontext2!="":
                results=results.replace(gcontext2,"")
        tocontent=""
        lanflag="0"
        if check_hotword>0:
            results=checkhotword(userid,results,self.hotword,self.lock)         
        if len(results)>=3 and tolan!="":
            tocontent,lanflag=llamacpp_hz2en(results,fromlan,tolan) #
        if savevoice==1 and ((savenoice>0 and results=="") or (results!="" and IsContainChinese(results)==False)):#yhy2026-03-18 没识别结果或者结果不是汉字的保存下来看看为啥
            folder_path=voice_path+"/"+time.strftime("%Y-%m-%d",time.localtime(start))
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            filename=folder_path+"/voice_"+userid+"_"+time.strftime("%Y-%m-%d-%H-%M-%S_",time.localtime(start))+str(count)+".wav"
            Save2Wav(audio,filename) 
            log.error("userid=%s,count=%07d,filename=%s,no hz!,text=[%s]",userid,count,filename,results)
        elif savevoice>1:#都保存
            folder_path=voice_path+"/"+time.strftime("%Y-%m-%d",time.localtime(start))
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)        
            filename=folder_path+"/voice_"+userid+"_"+time.strftime("%Y-%m-%d-%H-%M-%S_",time.localtime(start))+str(count)+".wav"
            Save2Wav(audio,filename) 
        asrlanguage=language2lan(asrlanguage)#yhy2026-05-29 
        result="{\"result\":\""+results+"\",\"errCode\":\""+errCode+"\",\"trans\":\""+tocontent+"\",\"lan\":"+lanflag+",\"language\":\""+asrlanguage+"\"}"       
        #log.warning("userid=%s,count=%07d,result=%s",userid,count,result)
        return result    
 
def server(license):    
    tmp=GetConfigParameter("PORT","9375","./asr.conf")
    httpport=int(tmp) 
    filename=GetConfigParameter("FILENAME","1.wav","./asr.conf")  
    ASR_MODEL_PATH=GetConfigParameter("ENGINE","./model","./asr.conf")
    model = MegaASR(
            model_path=ASR_MODEL_PATH + "/Qwen3-ASR-1.7B",
            lora_dir=ASR_MODEL_PATH+ "/mega-asr-merged",
            router_checkpoint=ASR_MODEL_PATH + "/audio_quality_router/best_acc_model.safetensors",
            routing_enabled=True,
            quality_threshold=0.0,
            device_map=None,
            keep_delta_on_gpu=True,
            backend="transformers",
        )
    print("----------------------------------------------")
    log.error("load mode ok,ASR_MODEL_PATH=%s,license=%d",ASR_MODEL_PATH,license)
    start = time.time()
    text=""
    language=""
    for i in range(2):   
        filename="./"+str(i)+".wav"
        with open(filename, 'rb') as file:
            audio = file.read()    
        if audio[0]==0x52 and audio[1]==0x49 and audio[2]==0x46 and audio[3]==0x46:
            pcm16=np.frombuffer(audio[44:],dtype=np.int16)
        else:
            pcm16=np.frombuffer(audio,dtype=np.int16)
        pcm16=pcm16.astype(float)*0.000030517578125
        if i<6:
            results = model.infer(audio=(pcm16, samples),return_objects=True, language="Chinese")
        else:
            results = model.infer(audio=(pcm16, samples),return_objects=True, language="English")
        print(results)
        text=""
        for i, r in enumerate(results): 
            text=text+r.text
            language=r.language
        print(text)
        print(language)
    end = time.time()      
    #print("asr use time: %6.3fs\n" %((end - start))) #rtf=35
    log.error("asr use time: %6.3fs,text=%s,lan=%s",(end - start),text,language)
    lock = threading.Lock()
    userid="test"
    hottext=text
    hotwords={}
    if check_hotword>0:
        gethotbymysql("",hotwords,lock)  
        start = time.time()
        hottext=checkhotword(userid,text,hotwords,lock)         
    end = time.time()      
    print("asr use time: %6.3f,result=%s\n" %((end - start),text))  
    print("asr use time: %6.3f,result=%s\n" %((end - start),hottext)) 
    
    # 运行服务器
    app=tornado.web.Application(handlers=[(r'/', Index),(r"/dotcasr", DoTcAsr,dict(model=model,license=license,lock=lock,hotword=hotwords))], autoreload=False, debug=False)
    # 手动生成server
    http_server = tornado.httpserver.HTTPServer(app,decompress_request=True)
    # 指定端口
    http_server.bind(httpport)
    # 开启多进程windows只能开一个
    http_server.start(1)
    print('HttpSvr port=%d,Ctrl+c quit!' % httpport)  
    log.error('HttpSvr start ok http port=%d,Ctrl+c quit!',httpport)
    # 开启
    try:
        tornado.ioloop.IOLoop.current().start()    
    except KeyboardInterrupt:
        print("CTRL+C")
    finally:
        print("HttpSvr finally...")        
        print("HttpSvr exit!")            
    return
     
if __name__ == '__main__':
    tmp=GetConfigParameter("LICENSE","0","./asr.conf")
    license=int(tmp)
    server(license)
    