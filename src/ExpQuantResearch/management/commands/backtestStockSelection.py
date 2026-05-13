import asyncio
import MySQLdb
import logging
import threading
import alpaca_trade_api
import finnhub
import numpy as np
import pandas_market_calendars as mcal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import pandas as pd
import pytz
from core.config import config
from django.core.management.base import BaseCommand
from datetime import date, datetime, timedelta
from core.symbolUtils import analyze_stocks, get_stock_data, get_stock_indicators
from core.dbUtils import clear_db
from core.utils import plot_standardized_data   
from core.stratTestUtils import get_stock_pct_change
log = logging.getLogger(__name__)

'''
to run test:
$python3 src/manage.py backtestStockSelection

1. get bullish, bearish, neutral markets
2. if bullish (for now), get symbols
3. take the symbols chosen and determine if price increased over the next x days 
4. get stats

Notes:
    - backtest done on Eastern timezone

NOTE:
    ideas and tests:
        - get percent usage of indicators in correct and incorrect models
            - remove useless indicators
        - use some math model to get proper trade range - how long does it take this info to affect price?
'''
class Command(BaseCommand):
    help = ''
    def handle(self, *args, **options):
        try:
            start_date = config.start_date
            end_date = datetime.today().date()
            # first get bullish/bearish/neutral results for each week
            url = config.alpaca.marketUrl
            api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

            spy_prices = {
                'date': [],
                'close': [],
            }
            response = api.get_bars(symbol='SPY', start=start_date, end=end_date, timeframe='1Day', feed='sip', sort='asc', limit=10000)
            log.debug(f"backtest stock selection: SPY data first index: {response[0].t} | final index: {response[-1].t}")
            for result in response:
                spy_prices['date'].append(result.t)
                spy_prices['close'].append(float(result.c))
            
            spy_df = pd.DataFrame(data=spy_prices)
            spy_df['date'] = pd.to_datetime(spy_df['date'])
            spy_df.set_index('date', inplace=True)

            test_start_date = datetime(year=2024, month=1, day=1, tzinfo=pytz.timezone('US/Eastern'))
            spy_df = spy_df[spy_df.index.weekday == 4]
            spy_df = spy_df[spy_df.index >= test_start_date]
            log.debug(f"backtest stock selection: spy df: {spy_df}")

            # get y-vars - if close increased x trading days later = 1, 0 otherwise
            market_statuses = [0] * len(spy_df['close'])
            timeframe = config.stocks.med_term_market.timeframe # temp for now
            market_pos_pct_change = config.stocks.med_term_market.pos_pct_change
            market_neg_pct_change = config.stocks.med_term_market.neg_pct_change
            step_size = int(timeframe / 5) # number of trading weeks 
            for i in range(len(spy_df['close']) - step_size):
                cur_spy_pct_change = (spy_df['close'].iloc[i + step_size] - spy_df['close'].iloc[i]) / np.abs(spy_df['close'].iloc[i])
                if cur_spy_pct_change > market_pos_pct_change:
                    market_statuses[i] = 1
                elif cur_spy_pct_change < market_neg_pct_change:
                    market_statuses[i] = -1

            spy_df['market_status'] = pd.Series(market_statuses, index=spy_df.index)

            # get list of symbols
            client = finnhub.Client(api_key=config.finnhub.apikey)
            mics = ['XNYS','XASE','BATS','XNAS']
            symbols = []
            for mic in mics:
                # get list of symbols
                try:
                    response = client.stock_symbols(exchange='US', mic=mic, currency='USD')
                except Exception:
                        log.error(f"error: unable to get symbols")
                        return
                for stock in response:
                    symbols.append(stock.get('symbol'))

            # remove any stocks with periods - these are substocks
            symbols = list(filter(lambda x: "." not in x, symbols))
            symbols = ['MOTG', 'VNM', 'GLBL', 'VXX', 'SMB', 'ATMP', 'MOAT', 'CBOE', 'SND', 'ICFI', 'NRDS', 'HUBG', 'ELVN', 'ALT', 'GOCO', 'HOOD', 'LEE', 'DTIL', 'ADEA', 'GCBC', 'PPHP', 'NEOG', 'OMCL', 'HRMY', 'CRCT', 'HMST', 'MNKD', 'NVNO', 'SLNHP', 'TTNP', 'BLKB', 'BCDA', 'TYRA', 'STRM', 'HALO', 'RILY', 'WEN', 'TRI', 'OBIO', 'OBLG', 'CGNT', 'KIDS', 'FFIN', 'PRCT', 'OXLCZ', 'BYRN', 'ENTG', 'SPT', 'LASR', 'EDBL', 'CTSH', 'PANW', 'DOCU', 'STAI', 'SPRY', 'IBRX', 'BOTJ', 'ROP', 'SRM', 'NETD', 'TMC', 'NCSM', 'EDIT', 'FMAO', 'ADI', 'OSW', 'NYXH', 'FCCO', 'DBX', 'SKYE', 'BFST', 'KRMD', 'TYGO', 'ZVSA', 'MNSBP', 'LIDR', 'ATLCZ', 'OCUL', 'FNWB', 'CAR', 'BBGI', 'EGBN', 'PPSI', 'RBCAA', 'ACB', 'HFBL', 'LAUR', 'TKNO', 'ADPT', 'VLGEA', 'RGEN', 'PECO', 'UONE', 'UDMY', 'XXII', 'ECOR', 'FCNCA', 'BAND', 'CERO', 'RICK', 'STRA', 'BFIN', 'ITRG', 'IBO', 'IHT', 'SENS', 'CMCL', 'CPHI', 'LCTX', 'LEGT', 'NRXS', 'TRX', 'PHGE', 'NEWP', 'IMO', 'PED', 'NBY', 'LSF', 'KNW', 'EVV', 'MPU', 'BRBS', 'IGC', 'TOVX', 'BCV', 'APT', 'VZLA', 'EP', 'AZTR', 'CVU', 'TMP', 'STXS', 'BURU', 'USAS', 'NG', 'DSS', 'HYLN', 'BKTI', 'REI', 'PLX', 'BRN', 'UAMY', 'IOR', 'BDL', 'NHC', 'MXC', 'CIK', 'SLND', 'CMT', 'ORLA', 'SEB', 'DHY', 'MTA', 'UMAC', 'NEN', 'OGEN', 'URG', 'EPM', 'INFU', 'NFGC', 'ECF', 'SOAR', 'MITQ', 'LGCY', 'VNRX', 'AIRI', 'FOXO', 'UAVS', 'NTIP', 'MYO', 'LODE', 'CVM', 'TMQ', 'ELMD', 'MGLD', 'MTNB', 'UEC', 'KULR', 'ACCS', 'INLX', 'CKX', 'UUU', 'PMNT', 'GROY', 'CET', 'XTNT', 'LGL', 'ACU', 'VGZ', 'EONR', 'ASM', 'TGB', 'PLAG', 'MHH', 'FURY', 'CTM', 'EFSH', 'ARMP', 'WRN', 'OPTT', 'SCCD', 'AMZE', 'PW', 'BTG', 'LEU', 'THM', 'SVT', 'SLI', 'TPET', 'SLSR', 'BHB', 'AMS', 'RVP', 'SCCF', 'VTAK', 'CLDI', 'DC', 'UUUU', 'IDR', 'AEF', 'PLG', 'EVI', 'INUV', 'XPL', 'ESP', 'CATX', 'DMYY', 'DIT', 'OBE', 'SACH', 'RLGT', 'FAX', 'GORO', 'OSTX', 'EIM', 'GBR', 'FSI', 'NGD', 'GENC', 'IE', 'WWR', 'SIF', 'CNL', 'EQX', 'BHM', 'OPHC', 'AIM', 'MAG', 'WYY', 'FSP', 'COHN', 'ARMN', 'MLN', 'VXZ', 'XMPT', 'SHYD', 'MOTI', 'MBIN', 'VLYPN', 'ACMR', 'KROS', 'MODV', 'TRNR', 'CMPR', 'SKYX', 'ESQ', 'STEP', 'PCVX', 'SNBR', 'XLO', 'KURA', 'VOR', 'SAVA', 'MRCY', 'LOAN', 'PLYA', 'GMGI', 'REGN', 'CUE', 'MNTK', 'XOMA', 'DLTR', 'NETDU', 'VTYX', 'OM', 'GIPR', 'DSP', 'HAIN', 'SILO', 'PAGP', 'FGBI', 'WAY', 'OCFC', 'BCML', 'LINK', 'GIFT', 'CYN', 'VSAT', 'ZD', 'BHRB', 'BTM', 'RGS', 'LINE', 'KRNY', 'FTCI', 'BPMC', 'XWEL', 'GOOD', 'REPL', 'SATL', 'LEXX', 'CZR', 'VOXR', 'FATE', 'IRBT', 'HQY', 'MGNI', 'CNFR', 'SAMG', 'NVDA', 'PRTH', 'FGBIP', 'DRCT', 'DAKT', 'AAON', 'SGA', 'CNTX', 'RILYP', 'LFUS', 'KZR', 'AUR', 'GITS', 'FCFS', 'GNLN', 'PYXS', 'DLTH', 'FORR', 'PXLW', 'JTAI', 'BTSG', 'AMSF', 'PNRG', 'GLSI', 'CRDL', 'STBA', 'KSCP', 'FXNC', 'CSGS', 'THRY', 'ADIL', 'PODD', 'UBCP', 'CHMG', 'TSRI', 'FLGT', 'MRVL', 'ALLO', 'LTRX', 'SELF', 'IART', 'EXE', 'SINT', 'ARDX', 'RCKT', 'TREE', 'LKQ', 'COMM', 'FNGR', 'CIGI', 'DRIO', 'CRTO', 'CLYM', 'SBCF', 'IRWD', 'NDSN', 'DMLP', 'SRTS', 'FRME', 'APLS', 'SLMBP', 'FPAY', 'PTMN', 'HOFT', 'KWE', 'RAIN', 'FWRG', 'RCKY', 'KTCC', 'TRUP', 'INCY', 'AZN', 'MARA', 'IMNN', 'MPB', 'ECBK', 'ORGO', 'CREX', 'KRYS', 'KSPI', 'IPW', 'CORT', 'PAYO', 'PNBK', 'BRKL', 'CRMT', 'FWONA', 'IPWR', 'CEVA', 'ABEO', 'PANL', 'IESC', 'ANDE', 'HELE', 'NTAP', 'WHLRD', 'MYRG', 'PAHC', 'MNSB', 'ARCB', 'TW', 'MGX', 'FORM', 'ATLC', 'TPIC', 'PRTA', 'PNFPP', 'TECX', 'XBIT', 'FFIC', 'ETON', 'NUVL', 'WHFCL', 'PASG', 'BJK', 'FSFG', 'DYAI', 'ADTX', 'NSYS', 'OBT', 'VALU', 'HSTM', 'VIVK', 'DENN', 'TLRY', 'ULH', 'CRGX', 'PNTG', 'SSRM', 'BWBBP', 'TRNS', 'CDXS', 'VRM', 'SPRO', 'FSBC', 'NAGE', 'LASE', 'SFIX', 'XFOR', 'ESPR', 'USIO', 'ALXO', 'ADTN', 'CDTX', 'TALK', 'FRPH', 'MCBS', 'IOBT', 'MKTX', 'TILE', 'MASS', 'SWKS', 'SGRY', 'COLL', 'CRIS', 'ISPO', 'ZVRA', 'LYRA', 'DXPE', 'EYE', 'POWW', 'WAFDP', 'PPH', 'IONS', 'PRLD', 'PI', 'NVCT', 'EOLS', 'WERN', 'NWL', 'NEO', 'MYPS', 'HCKT', 'ALGN', 'XPON', 'ZNTL', 'AURA', 'AVGO', 'GECCO', 'NEPH', 'REGCO', 'SRDX', 'INO', 'BNTC', 'EVO', 'MREO', 'FTAIN', 'INKT', 'MASI', 'SLAB', 'NSPR', 'TNYA', 'PLUR', 'ALBT', 'CMBM', 'CHRW', 'HLIT', 'ACNT', 'TRIP', 'MIDD', 'RMCF', 'FULTP', 'BHFAL', 'CMMB', 'WLFC', 'GDRX', 'AGEN', 'TARS', 'MRCC', 'DTST', 'KLTR', 'DAWN', 'PLAB', 'REGCP', 'HBANL', 'ICHR', 'ASRV', 'RSVR', 'TERN', 'AMBA', 'TRMK', 'CPSH', 'GAME', 'HBNC', 'PRAA', 'PRSO', 'CWD', 'SWKHL', 'MANH', 'RDIB', 'THRD', 'TBCH', 'AOUT', 'NGNE', 'LUCD', 'YORW', 'ATEC', 'AEP', 'NNBR', 'NERV', 'TLS', 'GECC', 'TFSL', 'LGVN', 'XNCR', 'MEIP', 'LEGH', 'MODD', 'BWIN', 'WYNN', 'TACT', 'RGTI', 'ARAV', 'ACXP', 'AXGN', 'VSTM', 'VIRC', 'PFG', 'UPLD', 'CVCO', 'ARCC', 'CTLP', 'LOPE', 'RDZN', 'BRLT', 'WING', 'BRAG', 'LYTS', 'WPRT', 'SNGX', 'CRWD', 'RIME', 'IRTC', 'TELA', 'BVFL', 'BRFH', 'DNTH', 'VERX', 'HTZ', 'ADUS', 'NXPI', 'THRM', 'AVBP', 'PALI', 'COHU', 'HBIO', 'INCR', 'HHS', 'GREEL', 'NTRSO', 'METC', 'TER', 'ADVM', 'NCMI', 'LIEN', 'NMIH', 'BKYI', 'OXLC', 'AEVA', 'FBNC', 'LNTH', 'ELBM', 'FBYD', 'CODA', 'CLBK', 'VTRS', 'PPIH', 'BHFAN', 'ASPI', 'OCSL', 'CPHC', 'OPINL', 'HNVR', 'GLTO', 'GLBZ', 'AMSC', 'TBRG', 'LPTH', 'SXTP', 'LSTR', 'FFAI', 'ODP', 'SNY', 'ENVB', 'MRIN', 'INVA', 'CPB', 'OLLI', 'VRME', 'VLY', 'LECO', 'TSHA', 'SCKT', 'RGP', 'AVDX', 'AVDL', 'SAIA', 'RAPP', 'REFI', 'MYSZ', 'HUT', 'NRXP', 'CCAP', 'CISO', 'METCB', 'FULT', 'SSP', 'MCVT', 'GCT', 'BRID', 'QRHC', 'CNTY', 'FARM', 'SMCI', 'CLST', 'PEP', 'SKIN', 'WHF', 'DAIO', 'KTOS', 'SMLR', 'GTI', 'ELTX', 'ASTE', 'OUST', 'TMUS', 'MYGN', 'ENGN', 'RPD', 'AOSL', 'MAYS', 'RELY', 'LCUT', 'LILA', 'HYMC', 'UAL', 'PZZA', 'TNGX', 'CDNA', 'STAA', 'NAII', 'STLD', 'FMBH', 'SOND', 'CACC', 'SOTK', 'CCB', 'CETY', 'CDT', 'FTEK', 'STKL', 'ORLY', 'CMCO', 'CCNEP', 'AIRT', 'TPCS', 'IHRT', 'GORV', 'FUSB', 'BBLG', 'HROW', 'OESX', 'XELB', 'CSCO', 'TRINI', 'UBX', 'TOMZ', 'PODC', 'AKTX', 'KAVL', 'FER', 'COYA', 'CVGW', 'IDYA', 'MGRX', 'BGC', 'HROWM', 'BIRD', 'SDST', 'HBANP', 'BIGC', 'SMSI', 'DBVT', 'BLBD', 'CYCN', 'AGNCN', 'SKYW', 'ATLCL', 'BELFB', 'ESGR', 'RBBN', 'OLMA', 'BMRC', 'EXPE', 'VOD', 'NEON', 'PCT', 'IVDA', 'GTLB', 'ACON', 'MORN', 'ISRLU', 'TRUE', 'AIMD', 'CRVO', 'RTH', 'GAIA', 'JACK', 'AEYE', 'TBLA', 'ESGRP', 'MULN', 'DHAI', 'BZFD', 'SNCY', 'ONBPP', 'STHO', 'AMRX', 'SANW', 'ROOT', 'WVVI', 'IPHA', 'AGRI', 'PROP', 'HAS', 'CSBR', 'HQI', 'CTXR', 'WAFD', 'PHUN', 'ALDX', 'XRX', 'RCMT', 'GLADZ', 'SHOT', 'MLAB', 'ASRT', 'CLMT', 'TAIT', 'GPRE', 'CLOV', 'TZOO', 'CPIX', 'CMTL', 'VMAR', 'GTBP', 'DIOD', 'AZTA', 'VNDA', 'NUKK', 'XOMAO', 'HST', 'MPWR', 'EPSN', 'KMB', 'MNRO', 'REFR', 'NVVE', 'INBS', 'VCEL', 'PRDO', 'RUSHB', 'GNFT', 'EU', 'PLMR', 'DXLG', 'FGEN', 'DCOM', 'KEQU', 'AMD', 'INSG', 'UNCY', 'BEAT', 'ISSC', 'LAB', 'AFJK', 'DIBS', 'FLNC', 'HCAT', 'SITM', 'FLL', 'APRE', 'QTRX', 'TBPH', 'PVBC', 'WEYS', 'EEFT', 'OPRX', 'SNCR', 'NDRA', 'SPTN', 'CNSP', 'DRVN', 'ABVX', 'MIGI', 'VERA', 'OSIS', 'BSY', 'IMG', 'NWSA', 'ACAD', 'PROV', 'RAVE', 'URBN', 'ONFO', 'MOFG', 'LTBR', 'INOD', 'SANM', 'WINA', 'MOLN', 'HBAN', 'PNFP', 'ESPO', 'RMBL', 'NYMTN', 'SSSSL', 'ENTO', 'JSPR', 'GOODN', 'NWBI', 'MTEX', 'RELI', 'SIDU', 'ONDS', 'FORA', 'TRML', 'MFIC', 'RDI', 'TFINP', 'KLXE', 'FTHM', 'ENVX', 'HLMN', 'BLNE', 'DHCNL', 'BTSGU', 'TCPC', 'WGS', 'PINC', 'IDAI', 'NRIM', 'ZION', 'MIRM', 'ATLX', 'SWBI', 'BSET', 'FKWL', 'GUTS', 'RPID', 'MBINN', 'ZLAB', 'GERN', 'AFJKR', 'EGAN', 'LIQT', 'ARQ', 'NAOV', 'ENPH', 'INTC', 'FBIOP', 'BASE', 'GBIO', 'AMKR', 'LITE', 'AYRO', 'RANI', 'ASO', 'LANDM', 'RWAYZ', 'QDEL', 'MNPR', 'CELU', 'COCH', 'TOI', 'BSLK', 'LQDT', 'TRIN', 'QCOM', 'LRHC', 'GABC', 'AKBA', 'HURC', 'TWIN', 'DALN', 'BOWN', 'CRDF', 'UROY', 'QURE', 'MGRM', 'AMIX', 'GURE', 'ANIX', 'ALTS', 'QETAR', 'DSGX', 'PCYO', 'MCFT', 'CGC', 'CHX', 'SGMT', 'COFS', 'AIRS', 'MLTX', 'INTZ', 'OAKUR', 'GOODO', 'NXPL', 'BKR', 'LDWY', 'RMTI', 'PLTK', 'REVB', 'RUSHA', 'GDEN', 'VERV', 'LAMR', 'ABUS', 'LANDP', 'EVGO', 'HWBK', 'BLMN', 'AEHR', 'AVXL', 'BSGM', 'ASPS', 'LE', 'OPOF', 'GSBC', 'TMCI', 'BPYPN', 'EXC', 'DIST', 'PDSB', 'UTMD', 'SPOK', 'AVPT', 'CGEM', 'CNOB', 'ABAT', 'ALZN', 'NVAX', 'PAYX', 'COLM', 'XCUR', 'III', 'PTCT', 'TRS', 'LFVN', 'DORM', 'INTS', 'IVP', 'FROG', 'GAINZ', 'CARV', 'ALCO', 'EDSA', 'LSEA', 'NVCR', 'CGTX', 'VC', 'XAIR', 'QSI', 'CCRN', 'RFIL', 'PAMT', 'FDBC', 'BLDE', 'ALGM', 'SGRP', 'RVYL', 'BBIO', 'SVC', 'IMCR', 'SONM', 'ASMB', 'ASTS', 'ABP', 'BLNK', 'GOOGL', 'CARG', 'APEI', 'AMWD', 'PHAR', 'EBMT', 'SFBC', 'MNTS', 'CRSP', 'CORZ', 'ABCL', 'OPK', 'FRAF', 'MKTW', 'WWD', 'INBK', 'FRMEP', 'SSKN', 'AIRJ', 'LNSR', 'ADAP', 'RAND', 'INTU', 'AREC', 'FRD', 'VLYPO', 'VRNA', 'ATRO', 'SCVL', 'PHIO', 'PTC', 'AKAM', 'LXRX', 'ARAY', 'NECB', 'ALNY', 'CVV', 'SBRA', 'FCAP', 'SEER', 'LGND', 'VNOM', 'FORD', 'PLUS', 'EBAY', 'SEDG', 'FGMCU', 'LOGI', 'NEXT', 'CZWI', 'AMST', 'RLMD', 'BENF', 'GOGO', 'ACNB', 'HFWA', 'LAND', 'ERII', 'OLB', 'BBH', 'NTNX', 'MCRB', 'WVVIP', 'ABL', 'VERU', 'CRWS', 'AIOT', 'CGBD', 'AKYA', 'NTRB', 'NSSC', 'KDP', 'ULTA', 'LWLG', 'CAPR', 'MERC', 'JOUT', 'FTNT', 'SBFM', 'AFJKU', 'RMR', 'CCCC', 'CTMX', 'FLEX', 'PDYN', 'FIVN', 'ISPR', 'LCNB', 'PEPG', 'INDP', 'TTSH', 'BRTX', 'RPAY', 'DHC', 'CZNC', 'LGO', 'GLDD', 'PGC', 'CLMB', 'PLCE', 'RUM', 'FBRX', 'VIR', 'HON', 'CLAR', 'FMNB', 'ANSS', 'PPC', 'BCRX', 'PTON', 'DPZ', 'SLDB', 'LVO', 'DRRX', 'TFIN', 'AQB', 'AUID', 'INTA', 'WOOF', 'NCNO', 'FEIM', 'CERT', 'RUN', 'UNIT', 'DOMO', 'OS', 'MBIO', 'NMRA', 'HAFC', 'FCUV', 'ASTC', 'BHFAO', 'VACHU', 'MFIN', 'EWBC', 'AERT', 'PRME', 'OXLCP', 'ABNB', 'CARM', 'SOHOB', 'AGNCM', 'CNTA', 'XRTX', 'PMTS', 'GP', 'IOSP', 'SYBT', 'SPFI', 'AGYS', 'CCLDO', 'POLA', 'SUNS', 'VTGN', 'PLXS', 'KRT', 'JRSH', 'ALTI', 'IAS', 'ATRC', 'KPLT', 'IVVD', 'ENSC', 'ROKU', 'SIGI', 'FSEA', 'OSBC', 'AMRN', 'PRPH', 'DMRC', 'RILYZ', 'OVLY', 'BEAM', 'ICUI', 'LPRO', 'KE', 'ZS', 'RCEL', 'CZFS', 'EXTR', 'MEDP', 'TSBK', 'COSM', 'ALMS', 'NVEC', 'FVCB', 'STSS', 'LBRDP', 'XMTR', 'WTW', 'UNTY', 'AGNCL', 'SEIC', 'NYMTZ', 'INBX', 'GH', 'CVLT', 'ARQT', 'AIP', 'TRVI', 'QNRX', 'OMER', 'FINW', 'CCOI', 'NEWTZ', 'ORRF', 'IRMD', 'VMD', 'LXEO', 'TTWO', 'EVER', 'GRPN', 'MRUS', 'ABSI', 'FWONK', 'AMAL', 'CSWCZ', 'UPBD', 'BPRN', 'PGEN', 'WKEY', 'LFWD', 'ONCY', 'LPTX', 'TSLA', 'IBCP', 'UMBFP', 'CLRB', 'IEP', 'AGNCP', 'UFCS', 'HNST', 'MRSN', 'CDW', 'ANNX', 'ATOM', 'WRAP', 'MGTX', 'HLNE', 'BTAI', 'SHYF', 'VRAR', 'CME', 'SABR', 'QLGN', 'BLDP', 'VLYPP', 'ALAB', 'OVID', 'PEBK', 'FRSH', 'HYFM', 'PHLT', 'QNTM', 'ADXN', 'TLF', 'WKSP', 'JRVR', 'LAZR', 'BAFN', 'ASUR', 'CBRL', 'CCCS', 'DUOT', 'LSAK', 'PLSE', 'QNCX', 'LFST', 'BBCP', 'OTRK', 'VRDN', 'SKWD', 'BCLI', 'HNNA', 'STRL', 'PLBC', 'NFLX', 'CDIO', 'CHRD', 'ALF', 'BRZE', 'OXBR', 'TGTX', 'CATY', 'DGXX', 'PCSA', 'LBTYA', 'SKYT', 'ATHA', 'OFLX', 'CNXN', 'FNLC', 'TNDM', 'TWFG', 'ACGL', 'NVA', 'SANG', 'RNAC', 'GIFI', 'AMRK', 'KINS', 'CGBDL', 'NWFL', 'SHC', 'KPRX', 'FTAI', 'LRFC', 'CMPX', 'AFRM', 'DRUG', 'TTMI', 'TSCO', 'NUWE', 'LINC', 'FDMT', 'NEWTI', 'VABK', 'SLNG', 'UGRO', 'HOFV', 'ACLX', 'BFRG', 'GLPI', 'TXN', 'GRNQ', 'PPBI', 'HROWL', 'HUMA', 'GEVO', 'IIIV', 'BANF', 'GLRE', 'NMTC', 'PIII', 'LULU', 'JAGX', 'LNKB', 'CROX', 'OCCI', 'AEMD', 'PLAY', 'PCTTU', 'ACHC', 'ERIE', 'TSBX', 'VCYT', 'LPSN', 'AWRE', 'EFSC', 'ZBIO', 'OKYO', 'VIGL', 'MGPI', 'AIRG', 'BLFS', 'TGL', 'JANX', 'EXEL', 'AENT', 'DOMH', 'BELFA', 'FIZZ', 'BYFC', 'GAINN', 'STRS', 'BNZI', 'BEEP', 'OXSQ', 'ESCA', 'AFBI', 'MDXG', 'VICR', 'ACRV', 'NVEE', 'VBTX', 'NAUT', 'HTCR', 'NWTG', 'TCX', 'ATLCP', 'ETNB', 'JBHT', 'DPRO', 'KOD', 'CVKD', 'RZLT', 'LCID', 'HLVX', 'BPYPP', 'AXTI', 'SHLS', 'EPRX', 'CING', 'CINF', 'MMLP', 'AXSM', 'SOPA', 'FATBP', 'ATOS', 'LARK', 'BMRA', 'ESHA', 'VANI', 'SDGR', 'MTRX', 'NWPX', 'APLT', 'LSBK', 'CDNS', 'SHBI', 'GAIN', 'AGFY', 'KPTI', 'SBFG', 'BGFV', 'FITBI', 'SMTC', 'BWFG', 'FCNCO', 'HWC', 'MEOH', 'HSIC', 'RILYN', 'INMB', 'NYMTM', 'QNST', 'NTCT', 'TNON', 'KFFB', 'LIVE', 'BOF', 'RKDA', 'RXRX', 'RAPT', 'ACT', 'WFRD', 'LIXT', 'LNT', 'VMEO', 'VTVT', 'LBRDA', 'BLFY', 'IMCC', 'RBB', 'CVGI', 'DLPN', 'GECCZ', 'LUNR', 'PBBK', 'PSEC', 'PKBK', 'PTIX', 'FIBK', 'REKR', 'SRZN', 'TVTX', 'NEWTG', 'OXLCO', 'LOCO', 'HRTX', 'MOGO', 'SEZL', 'RMNI', 'FA', 'MU', 'GTIM', 'CSWC', 'TTEK', 'QETA', 'BEEM', 'HWCPZ', 'EVTV', 'PRCH', 'MVIS', 'TRST', 'MDRR', 'BDTX', 'IBEX', 'RAIL', 'PVLA', 'ACLS', 'VEEE', 'AMZN', 'TEAM', 'AMPL', 'UBFO', 'AAOI', 'MNST', 'EWTX', 'BSBK', 'TCRT', 'CG', 'WHLR', 'GCMG', 'UPXI', 'OXSQG', 'GWAV', 'TASK', 'KELYB', 'OXLCL', 'WBTN', 'EVLV', 'RGLD', 'DTSS', 'LWAY', 'PENN', 'OPCH', 'TRINZ', 'SRCE', 'ALFUU', 'MGRC', 'TRMB', 'COIN', 'NXL', 'SPSC', 'SCYX', 'NTIC', 'CTNM', 'MRNA', 'AQMS', 'MNOV', 'KOSS', 'SERA', 'ARLP', 'IRON', 'BCBP', 'BFC', 'FEAM', 'MRM', 'VSEC', 'CODX', 'MIND', 'FTDR', 'RVMD', 'CPRX', 'ACIW', 'ATER', 'GTX', 'HDSN', 'OPBK', 'FISI', 'DJT', 'CVRX', 'ASNS', 'INDI', 'RILYT', 'GWRS', 'DVAX', 'INSE', 'LRCX', 'RDUS', 'EA', 'ALGS', 'RGLS', 'VVOS', 'FNKO', 'KALU', 'SOFI', 'ONBPO', 'USAU', 'HOVNP', 'WSBF', 'LTRY', 'BUSE', 'OCCIO', 'KHC', 'CWCO', 'FFIV', 'CCNE', 'IOVA', 'MRKR', 'ORGN', 'ASLE', 'BOWNR', 'REAL', 'BOWNU', 'SQFT', 'ABLLL', 'ALHC', 'EVCM', 'QTTB', 'CFFN', 'FBIZ', 'JAMF', 'IPAR', 'ATRA', 'QUIK', 'INZY', 'LLYVA', 'STRRP', 'WETH', 'CALC', 'CYCC', 'RYTM', 'SUPN', 'TPST', 'OPEN', 'INVE', 'ACTG', 'CRVS', 'YMAB', 'ARKR', 'JBSS', 'CFLT', 'NMFC', 'CCSI', 'DASH', 'VRNS', 'ASST', 'CABA', 'ZYME', 'EBC', 'SBGI', 'MVST', 'CLSK', 'MSBIP', 'MPAA', 'SGD', 'FHB', 'TUSK', 'AAPL', 'ALOT', 'AKRO', 'STTK', 'ATXG', 'QETAU', 'VERB', 'AGNCO', 'LESL', 'MATW', 'OPRT', 'PDFS', 'STI', 'CTKB', 'INNV', 'DUOL', 'TWST', 'SLNO', 'PTLO', 'TH', 'FTAIM', 'ARHS', 'CGON', 'CECO', 'CTNT', 'MSEX', 'RCAT', 'SRRK', 'SVRA', 'SYPR', 'VUZI', 'AVNW', 'CLNN', 'HSDT', 'APLD', 'LIF', 'POCI', 'IMMR', 'ZIMV', 'POOL', 'VSEE', 'UFPI', 'OSUR', 'MSBI', 'FYBR', 'INDB', 'SVCO', 'SATS', 'JAZZ', 'PGY', 'APGE', 'SYBX', 'RNXT', 'USLM', 'AGIO', 'HCWB', 'RXT', 'LOOP', 'ULCC', 'KALA', 'LVLU', 'EXFY', 'NATR', 'QLYS', 'PACB', 'FSBW', 'LLYVK', 'MXL', 'BBSI', 'NKTX', 'PEBO', 'CSTL', 'ADV', 'ERAS', 'FITB', 'GBDC', 'VKTX', 'VBNK', 'IPA', 'SIBN', 'MDXH', 'OMCC', 'NATH', 'INBKZ', 'RENT', 'WKHS', 'OXLCN', 'WTBA', 'DATS', 'GEOS', 'TIL', 'THTX', 'VFF', 'RKLB', 'CIVB', 'ORIC', 'DGII', 'BPOP', 'USCB', 'KALV', 'VXRT', 'ITRI', 'RARE', 'RLAY', 'PLPC', 'GSIT', 'TPG', 'ESSA', 'APYX', 'IAC', 'FFBC', 'ARTW', 'NCPL', 'FTFT', 'UFPT', 'VRTX', 'ASYS', 'COLB', 'AGNC', 'BTMD', 'MCHX', 'AMGN', 'FSLR', 'CBSH', 'HYPR', 'ROCK', 'EXAS', 'SGMA', 'PLTR', 'DKNG', 'FGMC', 'COEP', 'BCPC', 'EWCZ', 'JAKK', 'OPI', 'PFIS', 'KNSA', 'EML', 'SOWG', 'GROW', 'DNLI', 'HBCP', 'CRAI', 'KRUS', 'CXAI', 'ROIV', 'KITT', 'POWWP', 'HIVE', 'HOTH', 'UBSI', 'FRGT', 'VALN', 'TTD', 'ISRL', 'HPK', 'DHCNI', 'GEGGL', 'NEUP', 'ECDA', 'NWS', 'TTGT', 'FNWD', 'SMMT', 'RPRX', 'PTEN', 'PGNY', 'AUUD', 'FBLG', 'EPIX', 'CNXC', 'INM', 'IMVT', 'ELUT', 'ERIC', 'ADSK', 'LFMDP', 'AMPH', 'ONEW', 'LFMD', 'GTEC', 'ANAB', 'COCO', 'MIST', 'IBKR', 'FSUN', 'NDLS', 'HSON', 'TBBK', 'FRHC', 'DXCM', 'FLYW', 'VECO', 'NFE', 'SOUN', 'LMFA', 'MVBF', 'HBANM', 'BTBT', 'IDN', 'SBET', 'AUPH', 'XGN', 'KIRK', 'RRBI', 'SAGE', 'ALKS', 'ZIONP', 'RMBS', 'LYEL', 'CPZ', 'CRDO', 'KYTX', 'SCHL', 'FRPT', 'PSNL', 'PAL', 'FOSLL', 'QIPT', 'PESI', 'AMLX', 'QMCO', 'CDZI', 'DMAC', 'SPWH', 'AREB', 'KRON', 'SERV', 'CCBG', 'CNDT', 'UMBF', 'BCYC', 'XOS', 'ZM', 'GREE', 'OCCIN', 'PWP', 'FTLF', 'NBTX', 'BHFAP', 'ACET', 'TCBX', 'FLGC', 'WFCF', 'WINT', 'PPTA', 'PRTC', 'GNSS', 'SSNC', 'HURN', 'WATT', 'FSV', 'CVBF', 'IBIO', 'DOOO', 'UCTT', 'HNNAZ', 'PWOD', 'VRCA', 'VS', 'NWE', 'MDB', 'GCTK', 'CWBC', 'IROQ', 'CLNE', 'PCB', 'SNDX', 'ANY', 'NFBK', 'PRTS', 'LBTYB', 'NTWK', 'HRZN', 'ANTX', 'CRNX', 'TSAT', 'SGML', 'MAMA', 'SANA', 'DCGO', 'VRRM', 'SLP', 'EFOI', 'HTLD', 'ILLR', 'CRVL', 'MBINM', 'MESA', 'BVS', 'LYFT', 'NBIX', 'BOLT', 'NVTS', 'BLZE', 'TIPT', 'LNW', 'IRDM', 'MRBK', 'NMFCZ', 'SYM', 'LBTYK', 'FSTR', 'OLED', 'RWAY', 'APPS', 'PROF', 'OSS', 'LMAT', 'TELO', 'PMN', 'USGO', 'GEN', 'SYNA', 'ON', 'IVA', 'OAKUU', 'NRC', 'GIII', 'GRWG', 'SOHO', 'WTFC', 'COO', 'WNEB', 'SNTI', 'FRST', 'MSTR', 'SAFT', 'FIVE', 'NTGR', 'WDC', 'RSLS', 'PFXNZ', 'ACTU', 'CRNC', 'REG', 'SONO', 'ZTEK', 'ERNA', 'HOPE', 'FMST', 'APOG', 'VYGR', 'WLDN', 'MSGM', 'VTSI', 'GNPX', 'IMUX', 'STKS', 'ETSY', 'EVOK', 'IPGP', 'EHTH', 'DYN', 'SOHOO', 'SFM', 'BRKR', 'RIGL', 'UTHR', 'ARTL', 'XENE', 'WBD', 'LFCR', 'MQ', 'INTG', 'ODFL', 'CERS', 'IBOC', 'SHFS', 'INAB', 'SMTK', 'IKT', 'UEIC', 'BLBX', 'TCRX', 'SOBR', 'FULC', 'LQDA', 'SCOR', 'NDAQ', 'AVT', 'DDOG', 'PDLB', 'CFBK', 'XHG', 'EGHT', 'VREX', 'TURN', 'WASH', 'METCL', 'SGHT', 'ELSE', 'CELC', 'FCNCP', 'RPTX', 'MCRI', 'SABS', 'AVGR', 'WSC', 'ADMA', 'CFFI', 'ELVA', 'ANGL', 'CAC', 'RIOT', 'MYFW', 'ORKA', 'TXMD', 'AAME', 'META', 'TLN', 'LIND', 'LPCN', 'ENSG', 'VRNT', 'CHRS', 'VEEA', 'IRD', 'CPSS', 'BTCS', 'RMBI', 'XERS', 'DCOMP', 'PARA', 'BCTX', 'MRVI', 'WSBC', 'MDIA', 'ZUMZ', 'ACHV', 'CASY', 'ALGT', 'CARE', 'SENEB', 'XEL', 'CELH', 'SNEX', 'SCWO', 'PEGA', 'VVPR', 'XOMAP', 'AMCX', 'RDNT', 'EYEN', 'CLRO', 'CCEP', 'DAVE', 'FWRD', 'ASBP', 'BMRN', 'OCTO', 'ICMB', 'JJSF', 'ITRM', 'CCLD', 'TNXP', 'CMPS', 'ORMP', 'HGBL', 'APA', 'NUTX', 'OGI', 'NTLA', 'VERO', 'SMH', 'TROW', 'LUNG', 'BSRR', 'EVRG', 'VIAV', 'SLE', 'SWAG', 'RWAYL', 'APP', 'GALT', 'NOTV', 'ARVN', 'HFFG', 'UVSP', 'CLPT', 'SHIM', 'SNFCA', 'FAST', 'SIRI', 'LMNR', 'TXRH', 'MAPS', 'RILYG', 'MBRX', 'OVBC', 'LEDS', 'EQ', 'RBKB', 'SOHON', 'CETX', 'NTRP', 'ATXS', 'SNPX', 'GOOG', 'SMTI', 'NBTB', 'FLXS', 'NAVI', 'OTLK', 'MBCN', 'FBIO', 'RILYL', 'FLUX', 'DRS', 'LENZ', 'GILD', 'ACGLO', 'SFST', 'SRPT', 'MRAM', 'CAKE', 'JBLU', 'RIVN', 'DNUT', 'WHLRP', 'MCHP', 'CRUS', 'GSAT', 'NXT', 'NKSH', 'SLRC', 'LANC', 'ICAD', 'NXGL', 'BCAB', 'SYTA', 'ZURA', 'SWIM', 'ESOA', 'BDSX', 'COGT', 'PENG', 'MNMD', 'NXXT', 'SNDL', 'MAT', 'LILAK', 'INDV', 'KLIC', 'CBNK', 'SFNC', 'GECCI', 'CASS', 'ANIP', 'LZ', 'PLL', 'AXON', 'RDVT', 'WHLRL', 'PETS', 'FGI', 'BWB', 'RNAZ', 'RR', 'CIFR', 'ILPT', 'STOK', 'PRPO', 'TITN', 'PCH', 'SBAC', 'MSAI', 'DGICA', 'NXST', 'CTAS', 'WRLD', 'RILYK', 'JVA', 'NEWT', 'FORL', 'EXLS', 'APPF', 'JSM', 'RLYB', 'OTTR', 'SENEA', 'CRMD', 'CHTR', 'RNA', 'DERM', 'FUNC', 'FARO', 'LOVE', 'KRRO', 'GENK', 'AUBN', 'TCBIO', 'AFCG', 'BITF', 'WULF', 'WSFS', 'FOXF', 'EQIX', 'HTBK', 'BOKF', 'ARCT', 'ARWR', 'SYRE', 'SLRX', 'TCBK', 'PFX', 'INGN', 'NN', 'TEM', 'IDCC', 'INSM', 'MIRA', 'VERI', 'GSHD', 'SMID', 'COST', 'EXPI', 'STIM', 'GDYN', 'NB', 'PARAA', 'RELL', 'CDLX', 'GANX', 'CPRT', 'ESHAR', 'LSCC', 'DSGR', 'LMB', 'EBTC', 'CRSR', 'OFIX', 'TTEC', 'IZEA', 'ZG', 'EXPO', 'RRGB', 'BNGO', 'OKTA', 'HTO', 'ATNI', 'COOP', 'POWI', 'WMG', 'GTM', 'ROAD', 'ARTNA', 'FOLD', 'ADBE', 'PCAR', 'MTCH', 'HONE', 'TCBS', 'PROK', 'HOWL', 'DHIL', 'POWL', 'VCTR', 'ARRY', 'POAI', 'CURI', 'TLSI', 'ROST', 'FORLU', 'QRVO', 'HWKN', 'FOX', 'IDXX', 'ANGO', 'TECH', 'EFSCP', 'PMVP', 'SHEN', 'CENX', 'SHOO', 'RRR', 'EDUC', 'SMBC', 'THFF', 'OKUR', 'GLAD', 'HITI', 'KELYA', 'HWH', 'MAR', 'BETR', 'PUBM', 'ISTR', 'TXG', 'MITK', 'OMEX', 'YOSH', 'ZBRA', 'TRDA', 'SNSE', 'DXR', 'GO', 'TDUP', 'CENT', 'ECPG', 'LKFN', 'CHEF', 'PAVM', 'UNB', 'CRON', 'PSMT', 'CALM', 'FLNT', 'FENC', 'KTTA', 'BANX', 'PDEX', 'OCGN', 'IMMX', 'CLSD', 'HSII', 'VYNE', 'SSSS', 'TARA', 'CSX', 'BSVN', 'AVIR', 'FEMY', 'CELZ', 'PLRX', 'APPN', 'CNOBP', 'LPLA', 'KVHI', 'LAKE', 'AEIS', 'CTSO', 'NPCE', 'ANIK', 'ADN', 'PRGS', 'FHTX', 'ATAI', 'PET', 'IMKTA', 'KYMR', 'SIGIP', 'VRSK', 'CHCO', 'JKHY', 'WABC', 'BRY', 'PRTG', 'AROW', 'CSGP', 'EYPT', 'KLRS', 'ITIC', 'QCRH', 'JFBR', 'BIVI', 'MLKN', 'VITL', 'MELI', 'ACRS', 'BMEA', 'SIEB', 'GYRE', 'MGEE', 'SLNH', 'BNAI', 'NRIX', 'TCBI', 'HBT', 'MOVE', 'HNRG', 'CBFV', 'BL', 'LSTA', 'CSPI', 'ISRG', 'MBWM', 'STX', 'DWTX', 'SMPL', 'GEG', 'TLPH', 'ALLR', 'SGBX', 'ALKT', 'SLS', 'DCBO', 'FOXA', 'URGN', 'AVO', 'AMAT', 'CXDO', 'SGMO', 'CTBI', 'PBPB', 'SEAT', 'NYMT', 'ESTA', 'CBUS', 'TRAW', 'XPEL', 'ALRM', 'ALEC', 'CMCSA', 'COCP', 'HCSG', 'TCMD', 'BWEN', 'CYCCP', 'ICU', 'PATK', 'NHTC', 'AQST', 'CAAS', 'LIVN', 'PRPL', 'MRTN', 'PBHC', 'PHAT', 'RSSS', 'AGAE', 'LNZA', 'PAYS', 'DCTH', 'COKE', 'SEVN', 'NTRS', 'BJRI', 'PCRX', 'DSGN', 'ARKO', 'CYRX', 'DTI', 'MTSI', 'JCTC', 'CNFRZ', 'MSFT', 'EKSO', 'ISPC', 'JYNT', 'VLCN', 'CLDX', 'FANG', 'GNLX', 'ACIC', 'SHOP', 'CGNX', 'BTBD', 'REAX', 'LBRDK', 'BLIN', 'OXSQZ', 'IPSC', 'LGIH', 'PSTV', 'CLLS', 'ATNF', 'PBYI', 'NOVT', 'BHF', 'GOVX', 'CHCI', 'ICCC', 'ENTA', 'SSTI', 'WTFCP', 'GRAL', 'AVAV', 'IKNA', 'BOXL', 'STRT', 'PCTY', 'ARGX', 'CWST', 'BIOA', 'ELDN', 'ATLO', 'IGMS', 'MGYR', 'RXST', 'BJDX', 'KLAC', 'ADP', 'SLDP', 'STGW', 'GLUE', 'WBA', 'ALTO', 'MTVA', 'FAT', 'ONB', 'AEI', 'NCRA', 'PKOH', 'MDLZ', 'GRI', 'ALNT', 'RGNX', 'ULBI', 'SWTX', 'CTRN', 'ATEX', 'PRAX', 'THAR', 'BCAL', 'MLYS', 'FELE', 'OPAL', 'VRSN', 'ASTL', 'OTEX', 'TMDX', 'SBUX', 'ASTH', 'ACGLN', 'IRIX', 'BHFAM', 'AMED', 'GNTX', 'OLPX', 'IPDN', 'FLWS', 'FATBB', 'FITBO', 'POET', 'BKNG', 'LTRN', 'CMCT', 'PYPL', 'SONN', 'AVTX', 'CART', 'LRMR', 'STRR', 'MGNX', 'OPXS', 'FONR', 'TENB', 'RGCO', 'ATYR', 'DGICB', 'SDOT', 'ULY', 'SLM', 'UPWK', 'MDGL', 'OAKU', 'SIGA', 'WDFC', 'HOOK', 'GOSS', 'DWSN', 'SGC', 'BOOM', 'FCEL', 'KGEI', 'SCSC', 'IMRX', 'CHY', 'RVSB', 'ILMN', 'WTFCM', 'NSIT', 'EOSE', 'CNVS', 'BANR', 'VRA', 'MBUU', 'WSBCP', 'SSBK', 'NKTR', 'LIN', 'CADL', 'JMSB', 'UPST', 'FITBP', 'SQFTP', 'KOPN', 'PBFS', 'PTGX', 'STRO', 'MDBH', 'GPCR', 'BYND', 'XRAY', 'SCPH', 'GT', 'ANGI', 'BMBL', 'ONC', 'AHCO', 'RAMP', 'BKD', 'PINS', 'AGI', 'SPMC', 'HNI', 'RA', 'MGRB', 'CDP', 'YUM', 'KO', 'UWMC', 'DPG', 'MAGN', 'OFG', 'U', 'ZTR', 'GTY', 'ROK', 'TTI', 'EICA', 'KNF', 'DRH', 'WK', 'WD', 'TU', 'WST', 'MEG', 'HIPO', 'TY', 'SKX', 'RLTY', 'RVLV', 'IIPR', 'BOX', 'FBK', 'EQH', 'INGR', 'CSV', 'HXL', 'PBH', 'CRI', 'PRA', 'ECL', 'AMWL', 'BCE', 'CCO', 'CHD', 'IVR', 'MSM', 'MKC', 'PD', 'BY', 'CVS', 'BFS', 'HPQ', 'PG', 'SGU', 'GM', 'CUK', 'AZZ', 'EQC', 'RITM', 'ADM', 'DXC', 'SUN', 'DOCS', 'WOR', 'FSK', 'ATO', 'MTRN', 'ACEL', 'FCRX', 'ADC', 'SKT', 'SRV', 'CAPL', 'CURV', 'SGI', 'MUX', 'FTS', 'MDU', 'NVS', 'FSLY', 'DOCN', 'PCOR', 'PDX', 'HFRO', 'BZH', 'GHC', 'AZEK', 'PSO', 'FBIN', 'PRI', 'LC', 'VHI', 'LRN', 'SWK', 'ONTF', 'VAL', 'JHG', 'CAH', 'SCD', 'SFB', 'GWW', 'SHW', 'ARE', 'WH', 'HLX', 'GSBD', 'SPHR', 'MKL', 'BST', 'RBRK', 'PBI', 'YETI', 'BXP', 'FDP', 'CPRI', 'WU', 'IDA', 'VRT', 'MPX', 'IOT', 'SVV', 'CUZ', 'HASI', 'GEF', 'NWN', 'INN', 'ATEN', 'ASGI', 'OLP', 'QUAD', 'NCLH', 'MRK', 'NVT', 'DFIN', 'KMX', 'FBRT', 'TD', 'SPGI', 'GPC', 'SUP', 'KYN', 'PRKS', 'RCC', 'DUK', 'MLR', 'POST', 'COOK', 'PEB', 'CNP', 'UDR', 'DIS', 'ARLO', 'SCS', 'SCHW', 'PRU', 'CRBG', 'TEL', 'LUMN', 'JGH', 'TBN', 'MWA', 'USB', 'VMC', 'L', 'TRN', 'MTB', 'BGS', 'MOH', 'D', 'WTRG', 'PDT', 'BLE', 'VTLE', 'DELL', 'BANC', 'GPI', 'KVYO', 'BARK', 'BDC', 'CATO', 'BDX', 'DKL', 'WMS', 'AVB', 'PATH', 'USM', 'BFH', 'FT', 'FNV', 'AOS', 'ETN', 'ATGE', 'RSG', 'AIZN', 'USFD', 'A', 'TDW', 'BKE', 'MCD', 'JELD', 'CCL', 'WS', 'OLO', 'BBVA', 'WBS', 'IRM', 'SAND', 'WEX', 'FMC', 'ENS', 'UZE', 'BCS', 'MLI', 'SITE', 'INSW', 'HBI', 'BHLB', 'BHR', 'ORCL', 'CNX', 'EARN', 'LEO', 'CLB', 'TRP', 'G', 'CVEO', 'ASPN', 'CARR', 'AMC', 'JNJ', 'ARL', 'ACCO', 'TK', 'WLYB', 'EVTC', 'GRND', 'ALK', 'HOMB', 'RMD', 'HZO', 'AAP', 'MGY', 'GRX', 'AFGD', 'OXM', 'SCI', 'BP', 'GUG', 'LADR', 'TPR', 'PB', 'SPB', 'GETY', 'ST', 'KFS', 'PSTG', 'ECCX', 'DV', 'GEL', 'NABL', 'UPS', 'NXE', 'KEX', 'GPK', 'CLH', 'SMHI', 'TWLO', 'EQBK', 'CNK', 'CCRD', 'NPO', 'BFK', 'TRGP', 'SD', 'ACR', 'PMX', 'OIS', 'CRM', 'CNI', 'GRC', 'PHM', 'FRGE', 'GFF', 'BLD', 'SNA', 'AEG', 'FERG', 'MCS', 'RBOT', 'MITP', 'BEP', 'KSS', 'IMAX', 'CTO', 'APH', 'SKY', 'PFE', 'VLO', 'GBLI', 'MGRD', 'VET', 'CTRI', 'SMBK', 'DFH', 'MOV', 'ELME', 'TSE', 'EVN', 'NGL', 'EME', 'LEN', 'TJX', 'LPG', 'CQP', 'DCO', 'RCB', 'TRTX', 'KN', 'LHX', 'CLW', 'RGR', 'WOW', 'LMND', 'UAA', 'FIGS', 'HE', 'ENVA', 'FLUT', 'CARS', 'FL', 'MAA', 'GLW', 'ANVS', 'WHR', 'BKU', 'PKST', 'VICI', 'C', 'XOM', 'ARDT', 'MMI', 'LXFR', 'NMZ', 'BMEZ', 'ADCT', 'CSW', 'TDOC', 'TPB', 'WRB', 'PAR', 'AAT', 'VMI', 'RGA', 'SEI', 'DHI', 'NZF', 'PAXS', 'CBAN', 'SAT', 'OKE', 'NSC', 'DEI', 'RRC', 'SF', 'SBH', 'HUBS', 'WMB', 'WPM', 'SAR', 'SLQT', 'PAY', 'HBB', 'TEX', 'CNH', 'BTA', 'CRS', 'ADNT', 'LDP', 'LMT', 'HSY', 'ATR', 'ABBV', 'CBRE', 'GNK', 'MUR', 'PIM', 'LUCK', 'FCN', 'MFAN', 'AMTB', 'ED', 'ADT', 'EVH', 'MFAO', 'NCV', 'DAN', 'LYG', 'MBI', 'SOL', 'CIO', 'PLYM', 'CBT', 'TALO', 'UFI', 'MUSA', 'LW', 'BWNB', 'GXO', 'ESS', 'PMT', 'RF', 'NVR', 'AFGB', 'EMO', 'VEL', 'ATUS', 'FLO', 'FFWM', 'UBS', 'BOW', 'TRV', 'VSH', 'AWR', 'KBH', 'ASAN', 'CRL', 'KNTK', 'SYY', 'PX', 'IFF', 'CCJ', 'NEM', 'MFC', 'OPY', 'RACE', 'RDDT', 'RY', 'AIT', 'PRIM', 'KTF', 'ESE', 'VSCO', 'IDT', 'MHO', 'HCC', 'EPAM', 'IVT', 'BW', 'GRMN', 'KRO', 'TOST', 'SAH', 'HUM', 'LNN', 'HIX', 'AHR', 'AMRC', 'MCY', 'TE', 'ASGN', 'GBX', 'RWT', 'MC', 'SPRU', 'AHH', 'CACI', 'SLGN', 'TT', 'BWXT', 'BHK', 'NRT', 'WFC', 'AIG', 'AR', 'FPF', 'NOC', 'AX', 'KOS', 'TWI', 'RH', 'NKE', 'CTS', 'HTFB', 'ACRE', 'BWSN', 'CMSD', 'CAG', 'ALL', 'MQT', 'KD', 'LLY', 'BIP', 'PGRE', 'PHR', 'UIS', 'AXTA', 'NJR', 'GPN', 'BAM', 'MGM', 'GL', 'PLNT', 'BDN', 'TEAF', 'XPOF', 'THO', 'STT', 'WLKP', 'EG', 'TXT', 'MRC', 'OSK', 'HTFC', 'BURL', 'PRLB', 'DAY', 'MUA', 'APOS', 'LUV', 'BAC', 'ZETA', 'PINE', 'STZ', 'DTG', 'ACHR', 'GHY', 'RSI', 'J', 'ACM', 'SKE', 'B', 'AXP', 'KKR', 'MATX', 'EW', 'HWM', 'STN', 'RL', 'NBR', 'NGG', 'AVTR', 'RHI', 'RYAN', 'RPM', 'OEC', 'BME', 'EPR', 'PIPR', 'SNX', 'UNMA', 'HRI', 'UBER', 'AIZ', 'LYV', 'NSP', 'INFA', 'VTR', 'UHS', 'ABCB', 'NINE', 'EMN', 'AOMR', 'BBUC', 'FHI', 'DB', 'CPS', 'AHT', 'KGC', 'LYB', 'BSX', 'T', 'IAG', 'VTS', 'BGB', 'AVY', 'PFH', 'FCPT', 'HL', 'DTF', 'RYN', 'UNFI', 'ZWS', 'HES', 'EAT', 'DOC', 'DK', 'LII', 'TEF', 'GTES', 'KGS', 'NFG', 'ECAT', 'SHCO', 'EVRI', 'ARR', 'AWK', 'AWI', 'SAZ', 'SLB', 'AXS', 'FDX', 'TTC', 'OPFI', 'RNST', 'OWLT', 'ONIT', 'ACVA', 'CIEN', 'ARDC', 'NEUE', 'MSI', 'AMH', 'VCV', 'BTE', 'CNO', 'JOE', 'NGVT', 'RS', 'EFC', 'TYL', 'EXP', 'CMS', 'LSPD', 'THS', 'NE', 'BCSF', 'SAFE', 'O', 'HGTY', 'PDI', 'BNT', 'ARI', 'OBDC', 'EVEX', 'BFLY', 'HLI', 'MO', 'POR', 'TSN', 'MEI', 'ACA', 'KNSL', 'FSM', 'SOJE', 'AJG', 'VIRT', 'KTB', 'WY', 'SABA', 'AB', 'PSTL', 'CSR', 'ONTO', 'RM', 'GNRC', 'CL', 'BR', 'KW', 'CFG', 'ELF', 'SXT', 'HG', 'CXW', 'MDV', 'WHG', 'WT', 'NPK', 'PRMB', 'CHPT', 'PFS', 'DBRG', 'AGM', 'BK', 'TPL', 'ES', 'VVX', 'PDM', 'ULS', 'SILA', 'AME', 'UVE', 'SHO', 'ECCV', 'GOLF', 'PSN', 'THG', 'NRDY', 'PM', 'PACK', 'WTI', 'BYD', 'EVC', 'ALE', 'DHR', 'WNS', 'FNF', 'CALX', 'AL', 'MEC', 'INVH', 'PEN', 'WTS', 'RZC', 'GOOS', 'DNA', 'EPRT', 'BN', 'NVST', 'ALB', 'COLD', 'VST', 'HLT', 'TDS', 'ACI', 'AFGC', 'ASBA', 'TOL', 'PUK', 'RLJ', 'CMP', 'F', 'CLF', 'BRC', 'RDW', 'WCN', 'IPI', 'CVI', 'SPNT', 'PHX', 'MOS', 'CCI', 'HEI', 'TYG', 'EIG', 'UP', 'NREF', 'VTOL', 'RWTN', 'CSL', 'H', 'AFL', 'BNS', 'MSGS', 'TGT', 'OMF', 'SNOW', 'MYD', 'FBP', 'RGT', 'BLND', 'EXK', 'COP', 'CTEV', 'DSL', 'FTI', 'PFLT', 'GCI', 'CHWY', 'PRG', 'SW', 'WRBY', 'SAN', 'KEYS', 'DXYZ', 'RBA', 'FTK', 'SMR', 'UL', 'PLOW', 'CLS', 'KMI', 'TISI', 'GNTY', 'TRC', 'BILL', 'VHC', 'PSBD', 'SCL', 'RNR', 'WEC', 'PNW', 'MMU', 'SO', 'LAZ', 'KRG', 'NAD', 'RRX', 'TNET', 'PAYC', 'COF', 'WAB', 'FE', 'IQV', 'PSQH', 'TLYS', 'SSB', 'SITC', 'BRBR', 'PMTU', 'MYE', 'MSCI', 'PNR', 'WGO', 'BNL', 'RMAX', 'AON', 'UNH', 'LION', 'ING', 'ABG', 'WHD', 'HI', 'OPAD', 'W', 'RCD', 'BROS', 'LH', 'FLR', 'KBDC', 'LPX', 'WFG', 'CW', 'SMC', 'PLD', 'BSM', 'CE', 'BKH', 'BXMT', 'FRA', 'CR', 'AMR', 'NMCO', 'VATE', 'MG', 'CM', 'HUBB', 'EICC', 'NX', 'CAVA', 'CMA', 'APO', 'RC', 'BTT', 'PFGC', 'JEF', 'WDI', 'PCN', 'BOC', 'CVE', 'SDRL', 'SHAK', 'CODI', 'BX', 'SR', 'AUB', 'VKQ', 'CXT', 'RNGR', 'ALG', 'OI', 'EQT', 'AFGE', 'DNOW', 'BGSF', 'SRG', 'TFPM', 'ASB', 'SAP', 'FTV', 'GIL', 'HBM', 'SYK', 'COR', 'DSM', 'TBI', 'RVTY', 'CTRA', 'COTY', 'EB', 'UI', 'PH', 'WLK', 'TXNM', 'R', 'PRO', 'ATKR', 'GD', 'SPCE', 'TGNA', 'S', 'WSO', 'ASIX', 'APD', 'BLDR', 'MVT', 'AQNB', 'GRNT', 'JXN', 'HHH', 'MFM', 'ADX', 'DY', 'HP', 'TSQ', 'DE', 'BCC', 'JOBY', 'CIMN', 'OKLO', 'RZB', 'TFX', 'NNN', 'CMC', 'WMK', 'MGA', 'AORT', 'CMI', 'AQN', 'WAL', 'SAY', 'MSDL', 'CUBI', 'VRTS', 'HVT', 'GMS', 'RWTO', 'ANRO', 'NEU', 'MHK', 'FND', 'ESI', 'MA', 'NC', 'CGAU', 'LEG', 'REXR', 'XYL', 'SOC', 'NPFD', 'KREF', 'MMD', 'CF', 'KMPB', 'OVV', 'SYF', 'AVA', 'NEA', 'CNR', 'ANF', 'AMPX', 'CI', 'CLDT', 'ELAN', 'CP', 'SOR', 'VEEV', 'STEL', 'NPKI', 'STAG', 'GNL', 'CLVT', 'WWW', 'GCTS', 'LAD', 'ANET', 'FF', 'CDRE', 'GME', 'NMS', 'WKC', 'SJM', 'HOV', 'XHR', 'CTRE', 'GMED', 'FET', 'VNO', 'NBHC', 'FOA', 'GGZ', 'ALLY', 'HPP', 'NRK', 'JCI', 'NTST', 'MODG', 'OGN', 'PFN', 'SPG', 'GTN', 'EGY', 'JILL', 'FSNB', 'ORA', 'OGS', 'UTZ', 'RMM', 'GHM', 'LNG', 'CVX', 'YEXT', 'SMRT', 'TPC', 'MOD', 'STE', 'CBL', 'BBU', 'ICE', 'MP', 'AM', 'IONQ', 'AP', 'NEE', 'ROL', 'RFMZ', 'ITW', 'UE', 'ALC', 'LOB', 'OMI', 'JNPR', 'FNB', 'TDG', 'VNT', 'MATV', 'BXSL', 'CMSC', 'CION', 'WSM', 'BKT', 'USPH', 'BB', 'BAX', 'PK', 'NPCT', 'CRGY', 'GS', 'BBAI', 'NOTE', 'EBS', 'HOG', 'MQY', 'EXPD', 'AZO', 'HAE', 'KODK', 'MMC', 'TREX', 'CPAY', 'MPLX', 'MTD', 'ESNT', 'LNC', 'VZ', 'ELV', 'TFC', 'HUN', 'EVR', 'HAYW', 'IP', 'WTM', 'RTO', 'NRP', 'ALTG', 'PR', 'SON', 'NXDT', 'CAT', 'LOW', 'SXC', 'EPD', 'LBRT', 'CCK', 'BBY', 'RELX', 'MDT', 'FCX', 'GWH', 'WLY', 'DD', 'PKE', 'V', 'MTUS', 'PGR', 'ECCW', 'NOV', 'BMY', 'PFSI', 'CRH', 'DKS', 'BYON', 'UHT', 'MLNK', 'GIC', 'DLX', 'ODV', 'DNP', 'AIR', 'DLB', 'EL', 'CPT', 'HRL', 'DEC', 'FHN', 'SU', 'HOUS', 'QXO', 'THQ', 'EPAC', 'SDHY', 'EIX', 'PEO', 'MGR', 'MX', 'LXP', 'SNAP', 'UHAL', 'PRH', 'GHLD', 'QS', 'WPC', 'DBD', 'NL', 'ORC', 'BFAM', 'NI', 'TWO', 'CNC', 'KORE', 'REVG', 'NYC', 'GBCI', 'EBF', 'HTGC', 'PRM', 'AXL', 'ACN', 'OR', 'GCO', 'XPRO', 'ENB', 'WMT', 'JBL', 'UMH', 'LDI', 'MAS', 'GAP', 'MTDR', 'RBC', 'LOAR', 'MCK', 'AES', 'TGLS', 'SOJC', 'ECVT', 'CHGG', 'ITT', 'BEN', 'FOUR', 'BTI', 'CHCT', 'HQL', 'USAC', 'SPR', 'ESRT', 'STEM', 'HR', 'STWD', 'TKR', 'EMD', 'BBDC', 'NHI', 'PVH', 'IRT', 'FIX', 'OII', 'GVA', 'AFG', 'SXI', 'BCO', 'SBI', 'CDE', 'BBN', 'APG', 'UNP', 'KFRC', 'MAN', 'IPG', 'APAM', 'WELL', 'DDS', 'DG', 'LOCL', 'ECC', 'COMP', 'BRT', 'CIA', 'CRK', 'MHLA', 'ETWO', 'FAF', 'HIMS', 'AMCR', 'DEA', 'HCA', 'ZBH', 'NMAI', 'PSA', 'MFA', 'TNC', 'GIS', 'LAW', 'CNM', 'UTI', 'SDHC', 'MCB', 'KOP', 'VPG', 'XPO', 'AGX', 'BHP', 'ZVIA', 'SFBS', 'AKA', 'DVA', 'PMM', 'NET', 'CMTG', 'IBTA', 'UNF', 'INR', 'KIND', 'HMN', 'THW', 'AGL', 'NVRI', 'OTIS', 'CTOS', 'BA', 'KNX', 'BYM', 'HPE', 'AKR', 'DX', 'UZD', 'BIPC', 'BKKT', 'NLY', 'SRE', 'HGV', 'TMHC', 'FLG', 'PSX', 'CHN', 'SAM', 'GKOS', 'AMP', 'UGI', 'RCI', 'KAI', 'BCAT', 'PII', 'SKIL', 'AIN', 'FN', 'PDO', 'DOV', 'TGI', 'DHX', 'WES', 'SUI', 'GHI', 'CAL', 'RCUS', 'ECCC', 'DOW', 'UZF', 'EHAB', 'SRFM', 'CFR', 'GTLS', 'VOYA', 'ALSN', 'SKLZ', 'TAC', 'GES', 'RNG', 'ACV', 'USNA', 'AYI', 'MNR', 'GPMT', 'MAIN', 'AN', 'KAR', 'GLP', 'KWR', 'BWG', 'HIW', 'NRG', 'ODC', 'STM', 'SEM', 'CTVA', 'MHD', 'GROV', 'MEGI', 'OMC', 'GNE', 'SREA', 'JPM', 'LVS', 'SRL', 'DAR', 'GRBK', 'PCG', 'SEMR', 'PNC', 'ABM', 'TECK', 'BOH', 'BALY', 'AGO', 'SWX', 'EOG', 'BE', 'RBLX', 'SM', 'AMPY', 'BV', 'SAJ', 'CC', 'INSP', 'FPI', 'NCZ', 'EFX', 'CMG', 'CWK', 'BRDG', 'OBK', 'CYH', 'RIO', 'MET', 'FCF', 'CNQ', 'MVF', 'FUL', 'NPWR', 'AREN', 'VKI', 'BATL', 'IAUX', 'AEON', 'NAK', 'SSY', 'EMX', 'CYBN', 'GLDG', 'HUSA', 'ELA', 'SCCC', 'CIX', 'MLSS', 'SCCE', 'RCG', 'ZONE', 'DNN', 'GAU', 'PRK', 'SCCG', 'JOB', 'YCBD', 'REPX', 'GPUS', 'CCEL', 'CTGO', 'MAIA', 'EAD', 'SER', 'AUST', 'BLK', 'CHMI', 'AMT', 'RYI', 'SHEL', 'FR', 'TRAK', 'LTC', 'NTR', 'GFL', 'EEX', 'FG', 'CCIA', 'SLVM', 'BC', 'ASA', 'FTHY', 'COUR', 'ENR', 'RKT', 'FRT', 'GIB', 'VAC', 'FUBO', 'MLP', 'BALL', 'BWA', 'CXM', 'DUKB', 'CNS', 'ROG', 'NUVB', 'CLX', 'LCII', 'TPVG', 'ETD', 'EAF', 'CNNE', 'DEO', 'NYT', 'QTWO', 'ALV', 'HLN', 'DECK', 'CB', 'NOA', 'ONL', 'MITT', 'AMN', 'DCI', 'NEXA', 'BKN', 'KEY', 'PBA', 'TNL', 'AEO', 'THR', 'ALIT', 'MMM', 'AIV', 'PWR', 'CIM', 'AGS', 'RIV', 'NRGV', 'IEX', 'BRSP', 'MPC', 'AVNT', 'TSLX', 'AMBC', 'CUBB', 'MMS', 'AGCO', 'AEM', 'CPF', 'DT', 'JBTM', 'THC', 'PEG', 'REX', 'TRU', 'YELP', 'BHE', 'OUT', 'CCS', 'HII', 'BRO', 'CIVI', 'OWL', 'JLL', 'DTE', 'WNC', 'IBP', 'TEVA', 'IIIN', 'AIO', 'DGX', 'NIC', 'PAG', 'OGE', 'BMI', 'GWRE', 'RDY', 'WTTR', 'SG', 'TXO', 'PPG', 'CPNG', 'APLE', 'ALLE', 'MITN', 'M', 'COHR', 'WEAV', 'RLI', 'UAN', 'LDOS', 'BSTZ', 'ITGR', 'NSA', 'GDOT', 'AA', 'PJT', 'HLLY', 'HAL', 'NIE', 'ET', 'SPH', 'ARW', 'QSR', 'HY', 'KMT', 'MPW', 'HESM', 'FOR', 'DINO', 'ORI', 'BHC', 'SNDA', 'ECCF', 'RCL', 'CVNA', 'BMO', 'NXRT', 'GMRE', 'GNW', 'BIO', 'CCIF', 'RJF', 'KBR', 'PDS', 'NMG', 'FPH', 'RTX', 'DLY', 'JBI', 'BLCO', 'SKYH', 'CBU', 'MTH', 'SOJD', 'NVG', 'GDDY', 'SII', 'EICB', 'CNMD', 'WCC', 'BXC', 'PPL', 'AVNS', 'MAX', 'RVT', 'DDD', 'SLG', 'CNA', 'SLF', 'OC', 'AI', 'AEE', 'ABT', 'FSS', 'PBF', 'PARR', 'RPT', 'HTH', 'EQR', 'CABO', 'ZTS', 'PRS', 'MAC', 'EPC', 'LFT', 'HDB', 'STR', 'BBW', 'OOMA', 'NMI', 'JPI', 'EIC', 'WM', 'DNB', 'ERO', 'FC', 'APTV', 'RDN', 'IBM', 'LXU', 'VYX', 'IFN', 'HLF', 'CMSA', 'TMO', 'AWP', 'RIG', 'IGD', 'RFL', 'QBTS', 'ATS', 'PUMP', 'DTW', 'WIT', 'BBWI', 'SPIR', 'MIR', 'LEA', 'KRP', 'EGO', 'BTU', 'HTB', 'HLIO', 'WAT', 'UA', 'UNM', 'EGP', 'TG', 'CWT', 'PACS', 'DMB', 'TROX', 'HRTG', 'AVD', 'LEVI', 'YOU', 'BUD', 'ZIP', 'BIT', 'CPK', 'PL', 'HCXY', 'FMS', 'DAL', 'RES', 'FSCO', 'NGVC', 'BRX', 'FIS', 'OHI', 'FDS', 'MLM', 'DIN', 'INFY', 'MGRE', 'EHC', 'KR', 'GEO', 'NBXG', 'EFXT', 'GOF', 'IHG', 'MAV', 'MTW', 'NOW', 'BOOT', 'EXR', 'TIXT', 'SPXC', 'BG', 'PAAS', 'NUE', 'CHE', 'URI', 'GSK', 'SMG', 'ENOV', 'MTX', 'FICO', 'MYI', 'PMO', 'SSD', 'GBTG', 'TTE', 'CUBE', 'SEE', 'VRE', 'RHP', 'EMR', 'BODI', 'AXR', 'VVV', 'AMG', 'ELS', 'FUN', 'VFC', 'CBZ', 'MSA', 'CMU', 'DRI', 'MUE', 'MTG', 'CRC', 'HD', 'BRCC', 'RMI', 'LTH', 'WSR', 'SRI', 'SNDR', 'GE', 'CULP', 'IR', 'TFII', 'STC', 'ASH', 'MTZ', 'BNED', 'LZB', 'MBC', 'PRGO', 'SBSI', 'MS', 'GPOR', 'HIG', 'NOG', 'NNI', 'VNCE', 'EQS', 'KIM', 'FGN', 'ETR', 'DLR', 'UCB', 'OXY', 'HCI', 'FI', 'NGS', 'SIG', 'MCO', 'K', 'BEPC', 'DTB', 'DVN', 'PNNT', 'AC', 'NCDL', 'SNV', 'SA']
            log.debug(f"backtest stock selection: number of symbols: {len(symbols)}")

            # stat vars
            total_trades = 0
            total_successful_trades = 0

            # iterate throughout the dates and get symbols
            '''
            TODO: 
                new approach
                    - get all data at once in the format
                        - dict: key: symbol, val: df of all data
                    - once all data has been retrieved
                    - iterate through each date
                        - iterate through each symbol
                            - get data from before cut off date
                            - skip any data that dont meet meta feature requirements
                            - train models and make predictions
                '''
            # split list of symbols into a sub-list for each thread
            stock_selection_start_time = time.time()
            thread_sections = []
            section_size = 500
            for i in range(0, len(symbols), section_size):
                thread_sections.append(symbols[i : i + section_size])
            symbol_data_list = []
            # get data 
            with ThreadPoolExecutor(max_workers=len(thread_sections)) as thread_executor:
                futures = []
                for symbol_section in thread_sections:
                    # get_stock_data returns dict
                    futures.append(thread_executor.submit(get_stock_data, symbol_section))
                for future in as_completed(futures):
                    # result is a list of dicts
                    symbol_data_list.append(future.result())
            log.debug(f"time to get stock data: {(time.time() - stock_selection_start_time) / 60} minutes")

            test_symbols = []
            for dict in symbol_data_list:
                for symbol_key in dict.keys():
                    test_symbols.append(symbol_key)
            log.debug(f"stocks available for analysis: {test_symbols}")

            # once symbols and data are chosen, get indicators
            stock_indicator_start_time = time.time()
            symbol_data_w_indicators = []
            with ThreadPoolExecutor(max_workers=len(symbol_data_list)) as thread_executor:
                futures = []
                for symbol_dict_section in symbol_data_list:
                    futures.append(thread_executor.submit(get_stock_indicators, symbol_dict_section))
                for future in as_completed(futures):
                    symbol_data_w_indicators.append(future.result())
            log.debug(f"time to get stock indicators: {(time.time() - stock_indicator_start_time) / 60} minutes")
            
            for i in range(len(spy_df)):
                cur_datetime = spy_df.index[i]
                log.debug(f"backtest stock selection: current date: {cur_datetime}")
                trades_this_week = 0
                successful_trades_this_week = 0
                
                if spy_df['market_status'][i] == 1:
                    predicted_market_data = {
                        'market_status': 'bullish',
                        'timeframe': config.stocks.med_term_market.timeframe,
                        'pct_change': config.stocks.med_term_market.pos_pct_change,
                    }
                    stock_selection_start_time = time.time()
                    chosen_symbols = []
                    with ThreadPoolExecutor(max_workers=len(symbol_data_w_indicators)) as thread_executor:
                        futures = []
                        for symbol_data_section in symbol_data_w_indicators:
                            futures.append(thread_executor.submit(analyze_stocks, symbol_data_section, predicted_market_data, cur_datetime))
                        for future in as_completed(futures):
                            chosen_symbols.append(future.result())
                    log.debug(f"time to choose stocks: {(time.time() - stock_selection_start_time) / 60} minutes")
                    # convert list of lists to a list
                    log.debug(f"{cur_datetime} chosen symbols: {chosen_symbols}")
                    chosen_symbols = [entry for sublist in chosen_symbols for entry in sublist]
                    log.debug(f"{cur_datetime} formatted chosen symbols: {chosen_symbols}")

                    # for each symbol, determine if price increased/decreased
                    for symbol, target_pct_change in chosen_symbols.items():
                        trades_this_week += 1
                        total_trades += 1

                        trade_time = (cur_datetime + timedelta(days=3)).replace(hour=11, minute=30, second=0, microsecond=0) # monday
                        expire_time = trade_time + timedelta(days=timeframe) 

                        # check if market is open and switch to next open day if closed
                        nyse = mcal.get_calendar('NYSE')
                        schedule = nyse.schedule(start_date=trade_time.replace(tzinfo=None).date(), end_date=trade_time.replace(tzinfo=None).date())
                        market_closed = schedule.empty
                        while market_closed:
                            trade_time = trade_time + timedelta(days=1)
                            schedule = nyse.schedule(start_date=trade_time.replace(tzinfo=None).date(), end_date=trade_time.replace(tzinfo=None).date())
                            market_closed = schedule.empty

                        schedule = nyse.schedule(start_date=expire_time.replace(tzinfo=None).date(), end_date=expire_time.replace(tzinfo=None).date())
                        market_closed = schedule.empty
                        while market_closed:
                            expire_time = expire_time + timedelta(days=1)
                            schedule = nyse.schedule(start_date=expire_time.replace(tzinfo=None).date(), end_date=expire_time.replace(tzinfo=None).date())
                            market_closed = schedule.empty
                        log.debug(f"backtest stock selection: format check: trade time: {trade_time} | expire time: {expire_time}")

                        # make 'trade' and determine if price increased/decreased by the atr multiplier
                        init_price = float()
                        expire_price = float()
                        url = 'https://data.alpaca.markets'
                        api = alpaca_trade_api.REST(key_id=config.alpaca.apikey, secret_key=config.alpaca.secret, base_url=url, api_version='v2')

                        response = api.get_bars(symbol=symbol, timeframe='30Min', feed='sip', sort='asc', start=trade_time.replace(tzinfo=None).date(), end=expire_time.replace(tzinfo=None).date(), limit=10000)
                        for result in response:
                            date = result.t.to_pydatetime()
                            if date == trade_time:
                                init_price = float(result.c)
                            if date == expire_time:
                                expire_price = float(result.c)

                        cur_stock_pct_change = (expire_price - init_price) / init_price
                        log.debug(f"backtest stock selection: {cur_datetime} - {symbol} - {timeframe} - bullish: target pct change: {target_pct_change} | actual pct change: {cur_stock_pct_change}")
                        if cur_stock_pct_change > target_pct_change:
                            successful_trades_this_week += 1
                            total_successful_trades += 1

                    # log stats
                    log.debug(f"backtest stock selection: {cur_datetime} successful trades this week: {float(successful_trades_this_week / trades_this_week)}")
                    log.debug(f"backtest stock selection: {cur_datetime} current total success: {float(total_successful_trades / total_trades)}")

                # percent of symbols that incrased/decreased
                # net income
            return
        except Exception as err:
            log.warning(f"error backtesting stock selection process: {err}", exc_info=True)


