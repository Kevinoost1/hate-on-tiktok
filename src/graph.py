import pandas as pd
from database import *
import json
from pathlib import Path
import seaborn as sns
import matplotlib; matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import f_oneway, pearsonr, shapiro, wilcoxon
from statsmodels.stats.multicomp import MultiComparison
from itertools import combinations
import zstandard as zstd
from exploratory import *
import numpy as np
from scipy.stats import gaussian_kde
from analysis import *

"""
Script contains the graph for the news and hate exposure (already felt like analysis.py was getting too full)

DISCLAIMER: Probably the ugliest code you have ever layed eyes on. Sorry...
However, since it's only purpose is to graph something that is already there and does not include any logic that's important to the project, I could not really be bothered cleaning it up :(
The worst crime is probably seperating the news and hate by directly copying their respective parts instead of just passing an argument and putting it in a loop with [news, hate].
"""

BASE_DIR = Path(__file__).resolve().parent.parent

class Exposure_Graph:
    def __init__(self):
        self.db_path = BASE_DIR/'database/150_db.sqlite3'
        self.db = Database(db_path=self.db_path)

        following = Cross_Ref_Followers

        self.axesticks = 14
        self.fontsize_labels = 34
        self.fontsize_titles = 40
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d', 
                       'light_blue': '#3a68b0', 'red': '#c43a3a', 'light_green': '#7ab878', 'teal': '#2a8b7a', 'cream': '#f5f0e8',
                       'light_purple': '#d070c8', 'magenta': '#d040b0', 'violet': '#8a2080', 'deep_red': '#b83232', 'light_red': '#d98c8c',
                       'color_1_1': '#a8c4e8', 'color_2_1': '#c8dca0', 'color_3_1': '#e8a878', 
                       'eucalyptus': '#A5C8B8', 'yellow': '#E8CE91', 
                       'legacy': '#c8dca0', 'alt': '#83AFCB', 'niche': '#d070c8'}


    def get_exposure_news(self):
        query = f"""
        SELECT news_ratio, user_id
        FROM(
            SELECT AVG(
            CASE WHEN tc.topic = 'News, Politics & Societal Issues' THEN 1
            ELSE 0
        END) AS news_ratio, e.user_id AS user_id
        FROM engagements e
        LEFT JOIN transcript_classification tc ON tc.video_id=e.video_id
        WHERE tc.topic IS NOT NULL AND (e.user_id NOT IN (119, 132, 112, 147, 151) AND e.date_ymd >= "2023-10-01")
        GROUP BY e.user_id
        )
     
        """
        
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df

    def get_accounts(self):
        
        query = """
        SELECT AVG(legacy_ratio), AVG(alternative_ratio), AVG(niche_ratio)
        FROM(SELECT AVG(
            CASE WHEN a.unique_id IN ('rtvutrecht', 'omroeptilburg', 'npo3nl', 'dw_espanol', 'rtvenoticias', 'omroepeo', 'nieuwsuur', 'dwnews', 'tvpworld.com', 'rtvfocuszwolle', 'tv_oost', 'npostart', 'dw_the77percent', 'cbcsasknews', 'ardmediathek', 'sleutelstad', 'nos', 'rtvnof', 'dw.science', 'omroep_west', 'streekomroepwf', 'bbc', 'eenvandaag', 'france24_en', 'nhnieuws', 'max_telegraaf', 'omroeplwd', 'rtveen', 'dw.stories', 'nhhnieuws', 'ntrwetenschap', 'omroepbrabant', 'bbcnewsround', 'nosstories', 'tvwestug', 'regiortvconnect', 'bbcnews', 'nosjeugdjournaal', 'rtvdrenthe', 'aptnnews', 'omroephuman', 'rtvnoord', 'tvp3szczecin', 'omroepvenlonieuws', 'omroepwnl', 'noordhollands.dagblad', 'omroep_archipel', 'l1.limburg', 'newsnightbbc', 'orfnews', 'rtvpurmerend_official', 'telediario', 'cbcbecausenews', 'rijnmondsocial', 'studio.040', 'orfradiowienheute', 'br24', 'rtvoost', 'france24', 'at5.nl', 'omroepzeeland_', 'nosop3', 'omropfryslan', 'nosjournaalarabic', 'npr', 'omroep_flevoland', 'zdfinfo', 'boosbnnvara', 'omroepgld', 'vrt.nws', 'dw_planeta', 'cbcnews', 'ntromroep', 'boliviatv', 'rtenews', 'omroepvpro', 'rtvmaastricht', 'artetv1', 'swr', 'bollenstreekomroep', 'zdf')  
                THEN 1
            ELSE 0
            END) AS legacy_ratio,
            AVG(
            CASE WHEN a.unique_id IN ("dylan.page","ykfit","thierry_baudet","nowthisimapact","nederlandsepolitiek","couriernewsroom","oldskoolnews","thedailyshow","kamalahq","tagesschau"," nederlandviraal","officialbenderbij","yuvaltheterrible","kamergenoten_","meikemantel","denieuwsbv", "jimmydijk1985","daterightstuff","brutamerica","robertfkennedyjrofficial","jubilee","bastiaan_meijer", "tweedekamertiktoks","denhaagtube","fidias_eu_parliament","irisenthoven","hasandpiker", "latenightseth","forumvdemocratie","juisteantwoord","thecharliekirkshow","vicenews","thenewsmovement", "nlplan","thegoodliars","cestmocro","cestmocro2","cestmocro_comments","geenstijl.nl","www.geenstijl.nl", "trtworld","middleeasteye","eva.avrotros","pinknews","dannyrayes","ietsmetpils","pearlmania500", "jackmacbarstool","mirrorpolitics","dailymirror","xeviuniverse","valentivideos","janwillem_schut", "fallontonight","allian.p","de_standaard","bimsdepodcast","neurodivergent_nate","kamalaharris", "madeline_pendleton","harryjsisson","newsweek","localwinemum","xiandivyne","joysparkleshine", "allie_202_","vandaaginsidenl","advicefromlouisreloaded","yoyopipa","joshhelfgott","dezelfspodcast", "ongehoordnedtv","lbc","itsdeaann","cutbeans","vandaaginside","oog_groningen","lastweektonighthbo", "kafkanl","joris_explains", "dutchcrimenews.nl","santanamath","hopeyoufindyourdad", "gianmarcosoresi","ajplus","theparanormalfiles","politifun","groenlinkspvda","dashdobrofsky", "thierryslavendelplant","sander_tiktokt","parkergetajob","opps_______","newsnationnow","meidastouch", "heidireichinnek","dingnewscorp","cortimedia.nl","novaramedia","ivangtv","donlemon","democrats", "therecount","teamtrump", "spilnews", "jesse.klaver", "robjettend66", "lidewij.devos", "democraten66", "de3ekamer", "politie_eenheid_denhaag", "dekiesmannen", "politie", "gaabstemt", "politie_rotterdam", "dilanyesilgoz", "aaronparnas1", "philipdefranco", "partijvoordedieren", "charliekirkdebateclips", "progressief.nederland", "briantylercohen", "denk_nl", "rookvrije.generatie", "fryslanfvd", "bndestem", "bloombergpolitics", "courttvlive", "theipaper", "mustsharenews", "news.news.2025", "headquarters", "juliavgelder", "de_speld", "mdmotivator", "ipps0906", "joshlilj", "spencewuah",  "mothershipsg", "dumbbirchtree", "securityrendel", "waltermasterson", "jennevdwees", "subwaytakes",  "realraywilliam", "gewoonjorg", "matt.tv", "carterpcs", "sorry72636", "herkenjenl", "eohoedan", "schoppenvier",  "charlesnorthkorea", "aivideonederland", "jthefrog0", "spuitenenslikken", "noahglenncarter", "mshellly", "julia.mekkes",  "tmz", "sleiptok", "blauw.staat.aan", "johnny_blacksburg", "julielouise1975", "thenecessaryconversation", "martydoesnotplay",  "jeroenkroeze", "clipthecrisis", "sterrebrouwer", "ressepeasants", "parra_tv", "joeycontino2", "bbboptiktok", "andriessencp_",  "polarized_politics", "kabinetbaudet", "jos.dhaese", "veerle.romee", "thomasrpearson", "thotline_news", "studentenvoorleiden",  "davidpakmanshow", "politie_frans", "marcelbamberg", "jetten.hq", "insideedition", "bordotrey", "huffpost", "yourfavoriteguy", "whitehouse", "stephanvanbaarle",  "slopsta", "maxfostercnn", "da.ergin", "perezbrenna", "ruuddewild", "dutchbreakingnews", "adammockler", "savefamily24", "jadycadogan", "piersmorganuncensored",  "nederlandsche.democraten", "laurens.dassen", "kimberleyonline", "henri.bontenbal", "bnc_news", "studio_sedee", "mookspud", "kclmft", "bondirescue", "ugolord", "voicesecho", "stephaniesooyt", "scientology_audit", "dezelfspodcast", "feelthepodcasts", "d66leiden", "tjardastruik", "robandhaley", "murdertalknl",  "kies.vvd", "thatgoodnewsgirl", "svenvdmeulen", "rachel_mle", "nieuwscheckers", "politie_johan", "borderz.security", "davidjustinn", "justice4kokchan", "jordiwarners", "leftlogic.yamba", "gavinnewsom", "theverysimong", "robjettenlabubu", "darshanboerema_", "caroliennieuweboer", "zohran_k_mamdani", "ressevs", "openrotterdam", "nws.nws.nws", "nickshirleyy", "neurodivergent_nate", "nederlandpolitiek", "clarksonlawson", "austin.defies", "tvsensatiesnl", "remcocoerman", "reinoutz", "parentsontrial", "linedewitte", "chroniclesofolivia", "leftlaser", "dailywire", "gbnews", "thespectatormagazine", "omroeppownedofficial", "dailymail", "dailymail_us", "dailymailau", "dailymailfrontline", "dailymailnl", "dailymailuk", "dailymailsport", "dailymail.crime", "dailymailpolitics", "dailymailsportusa", "dailymailstreettalk", "news.com.au", "usatoday", "nypost", "washingtonpost", "washingtonpostlive", "foxnews", "fox4newsdallasfortworth", "fox9mn", "livenowfox", "nflonfox",  "msnbc", "msnow", "telemundo48elpaso", "wnytnewschannel13", "maddowshow", "newyorker", "scottishdailyrecord", "vox", "hollywoodreporter", "theeconomist", "halktv", "thesun", "talktv", "hartvannederland", "libelletv", "shownieuwsnl", "funxfm", "slamofficial", "radio538", "radio_veronica", "lbc", "sbs6nl", "rtlboulevard", "rtltalkshow", "nytopinion", "metrouk", "20minutesfrance", "bfmtv", "bangkokpost.official", "kron4.com", "lefigaro", "lemonde_en", "newstalkfm", "noticiastelemundo", "phillyinquirer", "puls24news", "robtv.be", "teletrece", "thairath_news", "tv_asahi_news", "vnexpress.official", "watson_news",  "dit.eo", "nytimes", "latimes", "theldnstandard", "thetelegraph", "wtopnews", "thetimes", "financialtimes",  "wallstreetjournal", "newstatesman", "cnn", "cnnee", "cnnindonesia", "cnnturk.com", "telegraaf.nl", "bnrnieuwsradio", "leeuwardercourant", "warringtonguardian", "ad_nl", "de_standaard", "de_volkskrant", "derstandard", "kyiv_post", "rheinischepost", "startribune", "9news", "c14news", "channel4", "channel4.0", "channel4comedy", "channel4dispatches", "channel4documentaries", "channel4entertainment", "channel4sport", "channel5", "channel5tiktok", "10newsau", "gmanews", "7newsaustralia", "7newsadelaide", "10newsplus", "12newsaz", "7newsmelbourne", "7newsq", "7newssydney", "8newsnow", "8world.news", "9newsadelaide", "9newsgoldcoast", "9newsmelbourne", "9newsperth", "9newsqueensland", "9newssydney", "newshubnz", "theprojecttv", "7newsaustralia", "7newsadelaide", "brisbanetimes", "tvlimburg", "nieuwsblad", "HetParool", "metronieuws", "c4news", "skynews", "cbsnews", "scrippsnews", "skynewsarabia", "skysports", "skysportsf1", "skysportsfootball", "wbrcnews", "wbztv", "wfsbnews", "winknewstv", "cbsmornings", "cbseveningnews", "48hourscbs", "cbschicago", "cbsla", "cbsmiami", "cbsnewyork", "cbsphiladelphia", "cbspittsburgh", "cbssports", "cbssundaymorning", "nfloncbs", "nbcsnl", "apnewsentertainment", "itvnews", "ctvnewstoronto", "ctvnewsvi", "yourmorningctv", "abcnews", "abcnewsaus", "abc7la", "abcnewslive", "abc7chicago", "abc7ny", "6abcactionnews", "abc", "abc.es", "abc13houston", "abc15arizona", "abc2020", "abc30_actionnews", "abc7newsbayarea", "abcindigenous", "abciview", "abcworldnews", "abscbnnews", "bostonglobe", "cnbc", "cnbci", "cnbcmakeit", "cnbrasil", "cp24breakingnews", "nbcnewyork", "nbc", "nbc10boston", "nbcnightlynews", "wfyipublicmedia", "cspanofficial", "nbcnews", "nbcnewyork", "staytunednbc", "nbcsouthflorida", "globalnews.ca", "thaipbs", "pbsdigitalstudios", "pbsnews", "thedailystar", "nottinghampost", "scotsman.newspaper", "independent", "independent_usa", "theirishindependent", "dailymirror", "dailyexpress", "dailytelegraph", "reuters", "newsweek", "nieuwsvandedag", "rtl.nl", "rtlnieuws.nl", "hln_be", "destentor", "euronews.al", "rtl.sport", "rtl419", "rtlaktuell", "rtldirekt", "rtlwest", "ntv.de", "sbsnews", "sbsrtlclips", "sbsnews_au", "aljazeeraenglish", "aljazeera", "aljazeera_mubasher", "straitstimes", "thenationalnews", "vtmnieuws_be", "ctvnews", "nrc_nl", "nu.nl", "gva", "eindhovensdagblad", "ouestfrance", "unreportedworld", "brabantsdagblad", "delimburger", "degelderlander", "euronews.tv", "demarkeronline", "bnnvara", "bnnvarakassa", "derodedraad.bnnvara", "zembla_bnnvara", "pointer.kroncrv", "npozapp", "avrotros", "radar.avrotros", "npostartnext", "kroncrv", "nossport", "nporadio1", "npo3fm", "bbccymruwales", "bbcearth", "bbcnewsbrasil", "bbcnewspidgin", "bbcradio2", "bbcradio4", "bbcsounds", "bbcthree", "bbcwomensfootball", "bbcsport", "vpro3voor12", "vpro_dvl", "vproclublees", "vprotegenlicht", "oogappels.bnnvara", "op1npo", "opgelicht.avrotros", "vroegevogels_bnnvara", "vrt1be", "vrtmax", "vrtnws.check", "omroepwestsport", "omroepmax", "omroepzwart", "ongehoordnederlandtv","npocultuur", "npokennis", "npoluister", "nporadio2", "deafspraak.vrtnws", "apnews", "cgtneurope", "channelnewsasia", "capemedianews", "guardian", "guardianaustralia", "zeit") 
                THEN 1
            ELSE 0
            END) AS alternative_ratio,
            AVG(
            CASE WHEN a.unique_id NOT IN ('rtvutrecht', 'omroeptilburg', 'npo3nl', 'dw_espanol', 'rtvenoticias', 'omroepeo', 'nieuwsuur', 'dwnews', 'tvpworld.com', 'rtvfocuszwolle', 'tv_oost', 'npostart', 'dw_the77percent', 'cbcsasknews', 'ardmediathek', 'sleutelstad', 'nos', 'rtvnof', 'dw.science', 'omroep_west', 'streekomroepwf', 'bbc', 'eenvandaag', 'france24_en', 'nhnieuws', 'max_telegraaf', 'omroeplwd', 'rtveen', 'dw.stories', 'nhhnieuws', 'ntrwetenschap', 'omroepbrabant', 'bbcnewsround', 'nosstories', 'tvwestug', 'regiortvconnect', 'bbcnews', 'nosjeugdjournaal', 'rtvdrenthe', 'aptnnews', 'omroephuman', 'rtvnoord', 'tvp3szczecin', 'omroepvenlonieuws', 'omroepwnl', 'noordhollands.dagblad', 'omroep_archipel', 'l1.limburg', 'newsnightbbc', 'orfnews', 'rtvpurmerend_official', 'telediario', 'cbcbecausenews', 'rijnmondsocial', 'studio.040', 'orfradiowienheute', 'br24', 'rtvoost', 'france24', 'at5.nl', 'omroepzeeland_', 'nosop3', 'omropfryslan', 'nosjournaalarabic', 'npr', 'omroep_flevoland', 'zdfinfo', 'boosbnnvara', 'omroepgld', 'vrt.nws', 'dw_planeta', 'cbcnews', 'ntromroep', 'boliviatv', 'rtenews', 'omroepvpro', 'rtvmaastricht', 'artetv1', 'swr', 'bollenstreekomroep', 'zdf')  
                AND a.unique_id NOT IN ("dylan.page","ykfit","thierry_baudet","nowthisimapact","nederlandsepolitiek","couriernewsroom","oldskoolnews","thedailyshow","kamalahq","tagesschau","nederlandviraal","officialbenderbij","yuvaltheterrible","kamergenoten_","meikemantel","denieuwsbv", "jimmydijk1985","daterightstuff","brutamerica","robertfkennedyjrofficial","jubilee","bastiaan_meijer", "tweedekamertiktoks","denhaagtube","fidias_eu_parliament","irisenthoven","hasandpiker", "latenightseth","forumvdemocratie","juisteantwoord","thecharliekirkshow","vicenews","thenewsmovement", "nlplan","thegoodliars","cestmocro","cestmocro2","cestmocro_comments","geenstijl.nl","www.geenstijl.nl", "trtworld","middleeasteye","eva.avrotros","pinknews","dannyrayes","ietsmetpils","pearlmania500", "jackmacbarstool","mirrorpolitics","dailymirror","xeviuniverse","valentivideos","janwillem_schut", "fallontonight","allian.p","de_standaard","bimsdepodcast","neurodivergent_nate","kamalaharris", "madeline_pendleton","harryjsisson","newsweek","localwinemum","xiandivyne","joysparkleshine", "allie_202_","vandaaginsidenl","advicefromlouisreloaded","yoyopipa","joshhelfgott","dezelfspodcast", "ongehoordnedtv","lbc","itsdeaann","cutbeans","vandaaginside","oog_groningen","lastweektonighthbo", "kafkanl","joris_explains", "dutchcrimenews.nl","santanamath","hopeyoufindyourdad", "gianmarcosoresi","ajplus","theparanormalfiles","politifun","groenlinkspvda","dashdobrofsky", "thierryslavendelplant","sander_tiktokt","parkergetajob","opps_______","newsnationnow","meidastouch", "heidireichinnek","dingnewscorp","cortimedia.nl","novaramedia","ivangtv","donlemon","democrats", "therecount","teamtrump", "spilnews", "jesse.klaver", "robjettend66", "lidewij.devos", "democraten66", "de3ekamer", "politie_eenheid_denhaag", "dekiesmannen", "politie", "gaabstemt", "politie_rotterdam", "dilanyesilgoz", "aaronparnas1", "philipdefranco", "partijvoordedieren", "charliekirkdebateclips", "progressief.nederland", "briantylercohen", "denk_nl", "rookvrije.generatie", "fryslanfvd", "bndestem", "bloombergpolitics", "courttvlive", "theipaper", "mustsharenews", "news.news.2025", "headquarters", "juliavgelder", "de_speld", "mdmotivator", "ipps0906", "joshlilj", "spencewuah",  "mothershipsg", "dumbbirchtree", "securityrendel", "waltermasterson", "jennevdwees", "subwaytakes",  "realraywilliam", "gewoonjorg", "matt.tv", "carterpcs", "sorry72636", "herkenjenl", "eohoedan", "schoppenvier",  "charlesnorthkorea", "aivideonederland", "jthefrog0", "spuitenenslikken", "noahglenncarter", "mshellly", "julia.mekkes",  "tmz", "sleiptok", "blauw.staat.aan", "johnny_blacksburg", "julielouise1975", "thenecessaryconversation", "martydoesnotplay",  "jeroenkroeze", "clipthecrisis", "sterrebrouwer", "ressepeasants", "parra_tv", "joeycontino2", "bbboptiktok", "andriessencp_",  "polarized_politics", "kabinetbaudet", "jos.dhaese", "veerle.romee", "thomasrpearson", "thotline_news", "studentenvoorleiden",  "davidpakmanshow", "politie_frans", "marcelbamberg", "jetten.hq", "insideedition", "bordotrey", "huffpost", "yourfavoriteguy", "whitehouse", "stephanvanbaarle",  "slopsta", "maxfostercnn", "da.ergin", "perezbrenna", "ruuddewild", "dutchbreakingnews", "adammockler", "savefamily24", "jadycadogan", "piersmorganuncensored",  "nederlandsche.democraten", "laurens.dassen", "kimberleyonline", "henri.bontenbal", "bnc_news", "studio_sedee", "mookspud", "kclmft", "bondirescue", "ugolord", "voicesecho", "stephaniesooyt", "scientology_audit", "dezelfspodcast", "feelthepodcasts", "d66leiden", "tjardastruik", "robandhaley", "murdertalknl",  "kies.vvd", "thatgoodnewsgirl", "svenvdmeulen", "rachel_mle", "nieuwscheckers", "politie_johan", "borderz.security", "davidjustinn", "justice4kokchan", "jordiwarners", "leftlogic.yamba", "gavinnewsom", "theverysimong", "robjettenlabubu", "darshanboerema_", "caroliennieuweboer", "zohran_k_mamdani", "ressevs", "openrotterdam", "nws.nws.nws", "nickshirleyy", "neurodivergent_nate", "nederlandpolitiek", "clarksonlawson", "austin.defies", "tvsensatiesnl", "remcocoerman", "reinoutz", "parentsontrial", "linedewitte", "chroniclesofolivia", "leftlaser", "dailywire", "gbnews", "thespectatormagazine", "omroeppownedofficial", "dailymail", "dailymail_us", "dailymailau", "dailymailfrontline", "dailymailnl", "dailymailuk", "dailymailsport", "dailymail.crime", "dailymailpolitics", "dailymailsportusa", "dailymailstreettalk", "news.com.au", "usatoday", "nypost", "washingtonpost", "washingtonpostlive", "foxnews", "fox4newsdallasfortworth", "fox9mn", "livenowfox", "nflonfox",  "msnbc", "msnow", "telemundo48elpaso", "wnytnewschannel13", "maddowshow", "newyorker", "scottishdailyrecord", "vox", "hollywoodreporter", "theeconomist", "halktv", "thesun", "talktv", "hartvannederland", "libelletv", "shownieuwsnl", "funxfm", "slamofficial", "radio538", "radio_veronica", "lbc", "sbs6nl", "rtlboulevard", "rtltalkshow", "nytopinion", "metrouk", "20minutesfrance", "bfmtv", "bangkokpost.official", "kron4.com", "lefigaro", "lemonde_en", "newstalkfm", "noticiastelemundo", "phillyinquirer", "puls24news", "robtv.be", "teletrece", "thairath_news", "tv_asahi_news", "vnexpress.official", "watson_news",  "dit.eo", "nytimes", "latimes", "theldnstandard", "thetelegraph", "wtopnews", "thetimes", "financialtimes",  "wallstreetjournal", "newstatesman", "cnn", "cnnee", "cnnindonesia", "cnnturk.com", "telegraaf.nl", "bnrnieuwsradio", "leeuwardercourant", "warringtonguardian", "ad_nl", "de_standaard", "de_volkskrant", "derstandard", "kyiv_post", "rheinischepost", "startribune", "9news", "c14news", "channel4", "channel4.0", "channel4comedy", "channel4dispatches", "channel4documentaries", "channel4entertainment", "channel4sport", "channel5", "channel5tiktok", "10newsau", "gmanews", "7newsaustralia", "7newsadelaide", "10newsplus", "12newsaz", "7newsmelbourne", "7newsq", "7newssydney", "8newsnow", "8world.news", "9newsadelaide", "9newsgoldcoast", "9newsmelbourne", "9newsperth", "9newsqueensland", "9newssydney", "newshubnz", "theprojecttv", "7newsaustralia", "7newsadelaide", "brisbanetimes", "tvlimburg", "nieuwsblad", "HetParool", "metronieuws", "c4news", "skynews", "cbsnews", "scrippsnews", "skynewsarabia", "skysports", "skysportsf1", "skysportsfootball", "wbrcnews", "wbztv", "wfsbnews", "winknewstv", "cbsmornings", "cbseveningnews", "48hourscbs", "cbschicago", "cbsla", "cbsmiami", "cbsnewyork", "cbsphiladelphia", "cbspittsburgh", "cbssports", "cbssundaymorning", "nfloncbs", "nbcsnl", "apnewsentertainment", "itvnews", "ctvnewstoronto", "ctvnewsvi", "yourmorningctv", "abcnews", "abcnewsaus", "abc7la", "abcnewslive", "abc7chicago", "abc7ny", "6abcactionnews", "abc", "abc.es", "abc13houston", "abc15arizona", "abc2020", "abc30_actionnews", "abc7newsbayarea", "abcindigenous", "abciview", "abcworldnews", "abscbnnews", "bostonglobe", "cnbc", "cnbci", "cnbcmakeit", "cnbrasil", "cp24breakingnews", "nbcnewyork", "nbc", "nbc10boston", "nbcnightlynews", "wfyipublicmedia", "cspanofficial", "nbcnews", "nbcnewyork", "staytunednbc", "nbcsouthflorida", "globalnews.ca", "thaipbs", "pbsdigitalstudios", "pbsnews", "thedailystar", "nottinghampost", "scotsman.newspaper", "independent", "independent_usa", "theirishindependent", "dailymirror", "dailyexpress", "dailytelegraph", "reuters", "newsweek", "nieuwsvandedag", "rtl.nl", "rtlnieuws.nl", "hln_be", "destentor", "euronews.al", "rtl.sport", "rtl419", "rtlaktuell", "rtldirekt", "rtlwest", "ntv.de", "sbsnews", "sbsrtlclips", "sbsnews_au", "aljazeeraenglish", "aljazeera", "aljazeera_mubasher", "straitstimes", "thenationalnews", "vtmnieuws_be", "ctvnews", "nrc_nl", "nu.nl", "gva", "eindhovensdagblad", "ouestfrance", "unreportedworld", "brabantsdagblad", "delimburger", "degelderlander", "euronews.tv", "demarkeronline", "bnnvara", "bnnvarakassa", "derodedraad.bnnvara", "zembla_bnnvara", "pointer.kroncrv", "npozapp", "avrotros", "radar.avrotros", "npostartnext", "kroncrv", "nossport", "nporadio1", "npo3fm", "bbccymruwales", "bbcearth", "bbcnewsbrasil", "bbcnewspidgin", "bbcradio2", "bbcradio4", "bbcsounds", "bbcthree", "bbcwomensfootball", "bbcsport", "vpro3voor12", "vpro_dvl", "vproclublees", "vprotegenlicht", "oogappels.bnnvara", "op1npo", "opgelicht.avrotros", "vroegevogels_bnnvara", "vrt1be", "vrtmax", "vrtnws.check", "omroepwestsport", "omroepmax", "omroepzwart", "ongehoordnederlandtv","npocultuur", "npokennis", "npoluister", "nporadio2", "deafspraak.vrtnws", "apnews", "cgtneurope", "channelnewsasia", "capemedianews", "guardian", "guardianaustralia", "zeit") 
                THEN 1
            ELSE 0
            END) AS niche_ratio
            FROM engagements e
            LEFT JOIN transcript_classification tc ON tc.video_id=e.video_id
            LEFT JOIN videos v ON v.id=e.video_id
            LEFT JOIN authors a ON a.id=v.author_id
            WHERE tc.topic = 'News, Politics & Societal Issues' AND (e.user_id NOT IN (119, 132, 112, 147, 151) AND e.date_ymd >= "2023-10-01")
            GROUP BY e.user_id
        )
        """
 
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df



    @staticmethod
    def load_channels(channelDefinitionFileName: str = BASE_DIR / 'src' / 'channel_def.json') -> tuple:
        with open(channelDefinitionFileName, 'r') as f:
            channels = json.load(f)

        return channels['official'], channels['unofficial'], channels['ordering']



    def get_exposure_monthly_news(self):
        query = f""" 
            WITH all_users AS (
                SELECT DISTINCT user_id FROM engagements WHERE user_id NOT IN (119, 132, 112, 147, 151)
            ),
            news_engagements AS (
                SELECT
                    e.user_id,
                    e.video_id,
                    e.date_ymd AS news_date
                FROM engagements e
                INNER JOIN transcript_classification tc ON tc.video_id = e.video_id
                WHERE tc.topic = 'News, Politics & Societal Issues'
                AND e.user_id NOT IN (119, 132, 112, 147, 151)
            ),
            user_timeframes AS (
                SELECT
                    a.user_id,
                    MIN(h.news_date) AS first_news_date,
                    MAX(h.news_date) AS last_news_date,
                    COUNT(h.video_id) AS total_news_videos
                FROM all_users a
                LEFT JOIN news_engagements h ON h.user_id = a.user_id
                GROUP BY a.user_id
            ),
            adjusted_start_dates AS (
                SELECT
                    user_id,
                    first_news_date,
                    last_news_date,
                    total_news_videos,
                    CASE
                        WHEN total_news_videos = 0 THEN NULL
                        ELSE MIN(
                            CASE
                                WHEN first_news_date < '2023-10-01' THEN '2023-10-01'
                                ELSE first_news_date
                            END,
                            last_news_date
                        )
                    END AS start_date
                FROM user_timeframes
            ),
            user_monthly_avgs AS (
                SELECT
                    user_id,
                    total_news_videos,
                    start_date,
                    last_news_date,
                    CASE
                        WHEN total_news_videos = 0 THEN 0
                        ELSE
                            total_news_videos * 1.0 /
                                CASE
                                    WHEN (julianday(last_news_date) - julianday(start_date)) / 30.44 = 0
                                    THEN 1
                                    ELSE (julianday(last_news_date) - julianday(start_date)) / 30.44
                                END
                    END AS monthly_avg_news
                FROM adjusted_start_dates
            )
            SELECT
                user_id,
                total_news_videos,
                start_date,
                last_news_date,
                ROUND(monthly_avg_news, 4) AS monthly_avg_news
            FROM user_monthly_avgs
            ORDER BY monthly_avg_news DESC;        
            """
 
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df


    def get_ratio_following_news(self):
        fol = Cross_Ref_Followers()
        user_engagements = fol.get_engagements(analysis='tc.topic="News, Politics & Societal Issues"')
        user_engagements = fol.merge_following_vs_non_following(user_engagements)
        df = fol.compute_ratios(user_engagements)

        return df

    def get_viewing_durations_news(self):
        data = DataExtract(self.db)
        feed_data = data.request_time_series()
        df = feed_data[0]

        df = df[df['topic'].notna()]
        df = df[df['new_session'] == 0]

        per_user = df.groupby('user_id').apply(lambda x: pd.Series({'news_med_wt': x.loc[x['is_news'] == 1, 'watch_time_s'].median(), 
                                                                   'no_news_med_wt': x.loc[x['is_news'] == 0, 'watch_time_s'].median()}))

        per_user['news_wt_med_diff'] = per_user['news_med_wt'] - per_user['no_news_med_wt']

        return per_user



    def get_exposure_ratio(self, threshold=5) -> pd.DataFrame:
        query = f"""
        SELECT hate_ratio, user_id
        FROM(
            SELECT AVG(
            CASE WHEN tc.intolerance >= {threshold} OR tc.impoliteness >= {threshold} THEN 1
            ELSE 0
        END) AS hate_ratio, e.user_id AS user_id
        FROM engagements e
        LEFT JOIN transcript_classification tc ON tc.video_id=e.video_id
        WHERE tc.topic = "News, Politics & Societal Issues" AND (e.user_id NOT IN (119, 132, 112, 147, 151) AND e.date_ymd >= "2023-10-01")
        GROUP BY e.user_id
        )
        
        """
        
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        print('Amount of users that did not see hate', df.loc[df['hate_ratio'] == 0, 'user_id'].nunique())
        print('Amount of users', df['user_id'].nunique())
        print('Percetage of users that did not see hate', df.loc[df['hate_ratio'] == 0, 'user_id'].nunique() / df['user_id'].nunique())

        return df

    def get_exposure_monthly(self, threshold=5) -> pd.DataFrame:
        query = f""" 
            WITH all_users AS (
                SELECT DISTINCT user_id FROM engagements WHERE user_id NOT IN (119, 132, 112, 147, 151)
            ),
            hate_engagements AS (
                SELECT
                    e.user_id,
                    e.video_id,
                    e.date_ymd AS hate_date
                FROM engagements e
                INNER JOIN transcript_classification tc ON tc.video_id = e.video_id
                WHERE tc.intolerance >= 5 OR tc.impoliteness >= 5
                AND tc.topic = "News, Politics & Societal Issues"
                AND e.user_id NOT IN (119, 132, 112, 147, 151)
            ),
            user_timeframes AS (
                SELECT
                    a.user_id,
                    MIN(h.hate_date) AS first_hate_date,
                    MAX(h.hate_date) AS last_hate_date,
                    COUNT(h.video_id) AS total_hate_videos
                FROM all_users a
                LEFT JOIN hate_engagements h ON h.user_id = a.user_id
                GROUP BY a.user_id
            ),
            adjusted_start_dates AS (
                SELECT
                    user_id,
                    first_hate_date,
                    last_hate_date,
                    total_hate_videos,
                    CASE
                        WHEN total_hate_videos = 0 THEN NULL
                        ELSE MIN(
                            CASE
                                WHEN first_hate_date < '2023-10-01' THEN '2023-10-01'
                                ELSE first_hate_date
                            END,
                            last_hate_date
                        )
                    END AS start_date
                FROM user_timeframes
            ),
            user_monthly_avgs AS (
                SELECT
                    user_id,
                    total_hate_videos,
                    start_date,
                    last_hate_date,
                    CASE
                        WHEN total_hate_videos = 0 THEN 0
                        ELSE
                            total_hate_videos * 1.0 /
                                CASE
                                    WHEN (julianday(last_hate_date) - julianday(start_date)) / 30.44 = 0
                                    THEN 1
                                    ELSE (julianday(last_hate_date) - julianday(start_date)) / 30.44
                                END
                    END AS monthly_avg_hate
                FROM adjusted_start_dates
            )
            SELECT
                user_id,
                total_hate_videos,
                start_date,
                last_hate_date,
                ROUND(monthly_avg_hate, 4) AS monthly_avg_hate
            FROM user_monthly_avgs
            ORDER BY monthly_avg_hate DESC;        
            """
 
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df

    def get_ratio_following(self):
        fol = Cross_Ref_Followers()
        user_engagements = fol.get_engagements(analysis='tc.impoliteness >= 5 OR tc.intolerance >= 5')
        user_engagements = fol.merge_following_vs_non_following(user_engagements)
        df = fol.compute_ratios(user_engagements)

        return df

    def get_viewing_durations(self):
        data = DataExtract(self.db)
        feed_data = data.request_time_series()
        df = feed_data[0]

        df = df[df['topic'] == 'News, Politics & Societal Issues']
        df = df[df['new_session'] == 0]

        per_user = df.groupby('user_id').apply(lambda x: pd.Series({'hate_med_wt': x.loc[x['high_hate'] == 1, 'watch_time_s'].median(), 
                                                                   'no_hate_med_wt': x.loc[x['high_hate'] == 0, 'watch_time_s'].median()}))

        per_user['hate_wt_med_diff'] = per_user['hate_med_wt'] - per_user['no_hate_med_wt']

        return per_user

   
    def create_exposure_news(self) -> plt.Figure:
        news_exposure_5 = self.get_exposure_news()["news_ratio"].dropna()
        news_monthly_5 = self.get_exposure_monthly_news()["monthly_avg_news"].dropna()
        following_5 = self.get_ratio_following_news()["is_following_ratio"].dropna()

        ratio_from_channel = self.get_accounts()

        viewing = self.get_viewing_durations_news()
        news_med_wt = viewing["news_med_wt"].dropna()
        no_news_med_wt = viewing["no_news_med_wt"].dropna()

        fig = plt.figure(figsize=(38, 18))

        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.30, wspace=0.30)
        fig.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.10)

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[0, 1])
        ax_d = fig.add_subplot(gs[1, 1])

        panels = [
            (
                ax_a, news_exposure_5, "news_ratio",
                self.colors["dark_blue"],
                "Percentage of Feed Containing News, Politics & Societal Issues (%)",
                True, 30, True,
            ),
            (
                ax_b, news_monthly_5, "monthly_avg_news",
                self.colors["orange"],
                "Average Number of News Videos per Month",
                False, 80, True,
            ),
        ]

        def plot_percentage_distribution(ax, values, color, xlabel, percent_x=False, bins=40, use_median=False):
            values = values[np.isfinite(values)].astype(float)

            if len(values) == 0:
                ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
                ax.set_ylabel("Percentage of User IDs", fontsize=self.fontsize_labels)
                return

            scale = 100 if percent_x else 1
            x = values * scale

            if np.min(x) == np.max(x):
                ax.hist(x, bins=[x[0] - 0.5, x[0] + 0.5], weights=np.full(len(x), 100 / len(x)), color=color, alpha=0.25, edgecolor=color)
                ax.axvline(x[0], color=color, linewidth=2.5, label=f"All users = {x[0]:.1f}")

            else:
                edges = np.histogram_bin_edges(x, bins=bins)
                bin_width = edges[1] - edges[0]

                sns.histplot(x=x, bins=edges, stat="percent", color=color, ax=ax, alpha=0.25, edgecolor=None, zorder=1)

                if len(x) >= 2 and np.std(x) > 0:
                    sns.kdeplot(x=x, color=color, ax=ax, linewidth=2.5, fill=False, bw_adjust=0.5, cut=0, zorder=2)

                    line = ax.lines[-1]
                    line.set_ydata(line.get_ydata() * bin_width * 100)
                    ax.fill_between(line.get_xdata(), line.get_ydata(), color=color, alpha=0.1, zorder=2)

                stat_x = np.median(x) if use_median else np.mean(x) 
                ax.axvline(stat_x, linestyle="--", color=color, linewidth=1.5, alpha=0.8,f"Median = {stat_x:.2f}" if use_median else f"Mean = {stat_x:.2f}"))

            ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
            ax.set_ylabel("Percentage of\nParticipants (%)", fontsize=self.fontsize_labels)

            ax.tick_params(axis="both", labelsize=self.fontsize_labels)
            ax.xaxis.set_major_locator(MaxNLocator(5))

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '' if x == 0 else f'{x:.0f}'))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            y_ticks = ax.get_yticks()
            y_max = max(y_ticks) if len(y_ticks) > 0 else 0
            ax.set_yticks([0, y_max])

            ax_d.legend(fontsize=self.fontsize_labels * 0.7, loc="upper right")

        for ax, values, _, color, xlabel, percent_x, bins, use_median in panels:
            plot_percentage_distribution(ax, values, color, xlabel, percent_x=percent_x, bins=bins, use_median=use_median)

        ax_a.set_xlim(0, 22)
        ax_a.set_ylim(0)

        ax_b.set_xlim(0, 1050)
        ax_b.set_ylim(0)

        plot_percentage_distribution(ax_c, following_5, self.colors["green"], "Percentage of News from Followed Accounts (%)", percent_x=True, bins=30, use_median=True)
        ax_c.set_xlim(0, 65)
        ax_c.set_ylim(0)

        channel_order = ["niche", "alternative", "legacy"]
        channel_labels = ["Niche", "Alternative", "Legacy"]

        channel_colors = [self.colors["niche"], self.colors["alt"], self.colors["legacy"]]
        channel_ratios = [
            ratio_from_channel["AVG(niche_ratio)"].iloc[0] * 100,
            ratio_from_channel["AVG(alternative_ratio)"].iloc[0] * 100,
            ratio_from_channel["AVG(legacy_ratio)"].iloc[0] * 100
        ] 

        y_positions = np.arange(len(channel_labels))

        for y, value, color in zip(y_positions, channel_ratios, channel_colors):
            ax_d.barh(y, value, height=0.60, facecolor=color, edgecolor=color, linewidth=1.5, zorder=2)
            ax_d.annotate(f"{value:.2f}%", xy=(value, y), xytext=(8, 0), textcoords="offset points", ha="left", va="center", fontsize=self.fontsize_labels, clip_on=False)

        ax_d.set_yticks(y_positions)
        ax_d.set_yticklabels(channel_labels, fontsize=self.fontsize_labels)

        ax_d.invert_yaxis()

        ax_d.set_xlim(0, 110)
        ax_d.set_xticks([0, 20, 40, 60, 80, 100])

        ax_d.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:.0f}%"))
        ax_d.set_xlabel("Average Percentage of News in Feeds (%)", fontsize=self.fontsize_labels)
        ax_d.tick_params(axis="both", labelsize=self.fontsize_labels)

        ax_d.spines["top"].set_visible(False)
        ax_d.spines["right"].set_visible(False)
        ax_d.spines["left"].set_visible(False)

        ax_d.margins(x=0.05, y=0.15)
        panel_axes = [ax_a, ax_b, ax_c, ax_d]

        for ax, label in zip(panel_axes, "abcd"):
            ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=self.fontsize_titles, fontweight="bold", ha="left", va="top")

        plt.show()

        return fig


    def create_exposure(self) -> plt.Figure:
        """
        Four-panel figure with three percentage histograms/KDE panels and
        a narrow right panel showing the per-user median watching-time
        difference (hate − non-hate) as a violin plot.
        """

        hate_exposure_5 = self.get_exposure_ratio()["hate_ratio"].dropna()
        hate_monthly_5 = self.get_exposure_monthly()["monthly_avg_hate"].dropna()
        following_5 = self.get_ratio_following()["is_following_ratio"].dropna()

        viewing = self.get_viewing_durations()
        hate_diff = viewing["hate_wt_med_diff"].dropna()
        hate_med_wt = viewing["hate_med_wt"].dropna()
        no_hate_med_wt = viewing["no_hate_med_wt"].dropna()

        fig = plt.figure(figsize=(38, 18))
        gs = fig.add_gridspec(3, 3, width_ratios=[3.25, 1, 0.3], hspace=0.5, wspace=0.4,)

        axes_left = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        ax_right = fig.add_subplot(gs[:, 1])

        fig.canvas.draw()

        for ax in [ax_right]:
            pos = ax.get_position()

            new_height = pos.height * 0.84
            ax.set_position([pos.x0, pos.y0, pos.width, new_height])

        left_left_x = axes_left[0].get_position().x0
        left_right_x = axes_left[0].get_position().x1
        right_left_x = ax_right.get_position().x0
        right_right_x = ax_right.get_position().x1
        
        panels = [
            (
                axes_left[0], hate_exposure_5, "hate_ratio",
                self.colors["purple"], "Percentage of News Feed Containing Hate (%)", True, 30, True,
            ),
            (
                axes_left[1], hate_monthly_5, "monthly_avg_hate",
                self.colors["orange"], "Average Number of Hate Videos per Month", False, 80, True,
            ),
            (
                axes_left[2], following_5, "is_following_ratio",
                self.colors["green"], "Percentage of Hate from Followed Accounts (%)", True, 30, True,
            ),
        ]

        def plot_percentage_distribution(ax, values, color, xlabel, percent_x=False, bins=40, use_median=False):
            values = values[np.isfinite(values)].astype(float)

            if len(values) == 0:
                ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
                ax.set_ylabel("% of User IDs", fontsize=self.fontsize_labels)
                return

            scale = 100 if percent_x else 1
            x = values * scale

            if np.min(x) == np.max(x):
                ax.hist(x, bins=[x[0] - 0.5, x[0] + 0.5], weights=np.full(len(x), 100 / len(x)), color=color, alpha=0.25, edgecolor=color)
                ax.axvline(x[0], color=color, linewidth=2.5, label=f"All users = {x[0]:.1f}")
            else:
                edges = np.histogram_bin_edges(x, bins=bins)
                bin_width = edges[1] - edges[0]

                sns.histplot(x=x, bins=edges, stat="percent", color=color, ax=ax, alpha=0.25, edgecolor=None, zorder=1)

                if len(x) >= 2 and np.std(x) > 0:
                    sns.kdeplot(x=x, color=color, ax=ax, linewidth=2.5, fill=False, bw_adjust=0.5, cut=0, zorder=2,)

                    line = ax.lines[-1]
                    line.set_ydata(line.get_ydata() * bin_width * 100)
                    ax.fill_between(line.get_xdata(), line.get_ydata(), color=color, alpha=0.1, zorder=2)

                stat_x = np.median(x) if use_median else np.mean(x)
                ax.axvline(stat_x, linestyle="--", color=color, linewidth=1.5, alpha=0.8, label=f"Median = {stat_x:.2f}" if use_median else f"Mean = {stat_x:.2f}")


            ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
            ax.set_ylabel("Percentage of\nParticipants (%)", fontsize=self.fontsize_labels)
            ax.tick_params(axis="both", labelsize=self.fontsize_labels)
            ax.xaxis.set_major_locator(MaxNLocator(5))

            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '' if x == 0 else f'{x:.0f}'))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            y_ticks = ax.get_yticks()
            y_max = max(y_ticks) if len(y_ticks) > 0 else 0
            ax.set_yticks([0, y_max])



        for ax, values, _, color, xlabel, percent_x, bins, use_median in panels:
            plot_percentage_distribution(ax, values, color, xlabel, percent_x=percent_x, bins=bins, use_median=use_median)

        axes_left[0].set_xlim(0, 13)
        axes_left[0].set_ylim(0)

        axes_left[1].set_xlim(0, 61)
        axes_left[2].set_xlim(0, 50)

        ax_right.set_title("", fontsize=self.fontsize_titles)

        parts = ax_right.boxplot([hate_med_wt.values, no_hate_med_wt.values], positions=[1, 2], widths=0.5, patch_artist=True, showmeans=False, showfliers=False)

        for box_color, box in zip([self.colors['red'], self.colors['light_red']], parts["boxes"]):
            box.set_facecolor(box_color)
            box.set_alpha(0.4)
            box.set_edgecolor(box_color)
            box.set_linewidth(2)
        for i, box_color in enumerate([self.colors['red'], self.colors['light_red']]):

            for line in [parts["caps"][i], parts["whiskers"][2 * i], parts["whiskers"][2 * i + 1]]:
                line.set_color(box_color)
                line.set_linewidth(2)

        for box_color, med in zip([self.colors['red'], self.colors['light_red']], parts["medians"]):
            med.set_color(box_color)
            med.set_linewidth(3)

        hate_median = hate_med_wt.median()
        no_hate_median = no_hate_med_wt.median()

        legend_hate = Line2D([], [], color=self.colors["red"], linewidth=2.5,
                             label=f"Hateful News Median = {hate_median:.2f}s")
        legend_no_hate = Line2D([], [], color=self.colors["light_red"], linewidth=2.5,
                                 label=f"Non-Hateful News Median = {no_hate_median:.2f}s")

        ax_right.set_ylim(0)

        ax_right.set_ylabel("Median Viewing Time per User (sec)", fontsize=self.fontsize_labels)
        ax_right.tick_params(axis="both", labelsize=self.fontsize_labels)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.legend(handles=[legend_hate, legend_no_hate], fontsize=self.fontsize_labels * 0.7, loc=(0.05, 1.065))

        ax_right.set_xticks([1, 2])
        ax_right.set_xticklabels(["Hateful\nNews", "Non-Hateful\nNews"], fontsize=self.fontsize_labels * 0.75)

        panel_labels = [axes_left[0], axes_left[1], axes_left[2], ax_right]
        for idx, ax in enumerate(panel_labels):
            label = 'abcd'[idx]
            x_pos = -0.10 if idx == 3 else 0.02
            y_pos = 1.20 if idx == 3 else 0.98
            ax.text(x_pos, y_pos, label, transform=ax.transAxes, fontsize=self.fontsize_titles, fontweight="bold", ha="left", va="top",)

        plt.show()
        return fig
    
    def export_png(self,  *figs: tuple[plt.Figure], custom_label='exposure') -> None:
        for i, fig in enumerate(figs):
            fig.savefig(f'plots/{custom_label}_{i}.png', dpi=300)


    def export_tiff(self, *figs: tuple[plt.Figure], custom_label='exposure') -> None:
        for i, fig in enumerate(figs):
            fig.savefig(f'plots/{custom_label}_{i}.tiff', format='tiff', dpi=600)


if __name__=='__main__':
    graph = Exposure_Graph()
    exposure_plot = graph.create_exposure()
    exposure_plot_news = graph.create_exposure_news()

    graph.export_png(exposure_plot, exposure_plot_news)
    graph.export_tiff(exposure_plot, exposure_plot_news)
