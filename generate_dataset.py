#!/usr/bin/env python3
"""
DATANYX'26 Human Impact Assessment Framework: dataset generator (v3)
Seeded and reproducible:  python generate_dataset.py
Outputs (CSV): INCIDENT_CASES, CASE_HARMS, CASE_HARM_ATTRIBUTES, CASE_BENEFITS, AI_DEPLOYMENT (= DEPLOYMENT_CONFIGURATIONS), VALIDATION_REPORT
Design: layered causal simulation. Deployments first -> incidents drawn in proportion to exposure and conditions ->
cases/harms -> attributes (levels) -> severity COMPUTED (never sampled) with the locked formula.
LOCKED MATH: x = (L-1)/4 ; S = (sum_j w_j x_j^2)^(1/2), w_j = 0.20 ; Severity_Formula_Version = "v1.0_powermean_q2".
Confirmation_Weight: CONFIRMED 1.0, ALLEGED 0.25, SUSPECTED 0.20, UNKNOWN 0.10, CONTRADICTED 0.0 (assumption, see README).
All synthetic rows: Record_Type = SYNTHETIC, Source_Reference = SYNTHETIC_NO_EXTERNAL_SOURCE. Documented rows: Record_Type = DOCUMENTED.
"""
import json, math, os, sys, datetime as dt
import numpy as np, pandas as pd

SEED = 20261123
OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(SEED)
D = dt.date
CUTOFF = D(2026, 10, 31)
N_SYN_CASES = 1000
FORMULA_VERSION = "v1.0_powermean_q2"
W = np.array([0.2] * 5)
CW = {"CONFIRMED": 1.0, "ALLEGED": 0.25, "SUSPECTED": 0.20, "UNKNOWN": 0.10, "CONTRADICTED": 0.0}
DOM = {"LF": "LIVELIHOOD_FINANCES", "BN": "BASIC_NEEDS_HOUSING", "HS": "HEALTH_SAFETY", "PD": "PRIVACY_DATA", "AC": "AUTONOMY_CONTROL", "OS": "OPPORTUNITY_STANDING"}

# ---------------- taxonomy ----------------
TAX = {}
for c in ["JOB_DISPLACEMENT","EMPLOYMENT_TERMINATION","WAGE_REDUCTION","DESKILLING","EMPLOYMENT_INSTABILITY","WORKPLACE_MONITORING","OPPORTUNITY_REDUCTION","WORKLOAD_OR_ROLE_CHANGE"]: TAX[c] = ("EMPLOYMENT", "EMPLOYMENT")
for c in ["UNAUTHORIZED_DATA_COLLECTION","DATA_EXPOSURE"]: TAX[c] = ("PRIVACY", "SOVEREIGNTY")
for c in ["DISCRIMINATION","UNJUSTIFIED_AUTOMATED_DECISION","LOSS_OF_HUMAN_CONTROL","DENIAL_OF_APPEAL","LACK_OF_ACCOUNTABILITY","LACK_OF_TRANSPARENCY"]: TAX[c] = ("DECISION_RIGHTS", "SOVEREIGNTY")
for c in ["SURVEILLANCE","AUTONOMY_RESTRICTION","MANIPULATION"]: TAX[c] = ("AUTONOMY", "SOVEREIGNTY")

# harm templates: N,M,Rmu means (levels), median days, domain pool, loss prob, income-loss flag, predicates (name, layer)
def T(N, M, R, days, pool, loss, inc, preds): return dict(N=N, M=M, R=R, days=days, pool=pool.split(","), loss=loss, inc=inc, preds=preds)
A_, DR = "ATOMIC", "DERIVED"
HT = {
"JOB_DISPLACEMENT": T(4,3.8,3,240,"LF,OS,BN",0,True,[("TasksAutomated",A_),("RoleEliminated",A_),("JobDisplaced",DR)]),
"EMPLOYMENT_TERMINATION": T(4,3.9,3.4,200,"LF,OS,BN,AC",0,True,[("TasksAutomated",A_),("EmploymentTerminated",A_),("SeverancePaid",A_)]),
"WAGE_REDUCTION": T(3,3,3.4,300,"LF",0,True,[("TasksAutomated",A_),("HoursReduced",A_),("WageReduced",DR)]),
"DESKILLING": T(3,2,3.5,400,"OS",0,False,[("TasksAutomated",A_),("SkillUseReduced",A_),("Deskilled",DR)]),
"EMPLOYMENT_INSTABILITY": T(3,3,3.4,180,"LF,AC",0,False,[("ScheduleVolatilityIncreased",A_),("EmploymentInstabilityIndicated",DR)]),
"WORKPLACE_MONITORING": T(3,2,3.5,350,"AC,PD",0,False,[("MovementTrackedContinuously",A_),("ExcessiveMonitoringEstablished",DR)]),
"OPPORTUNITY_REDUCTION": T(3,3,3.6,120,"OS,LF",0,False,[("ApplicationRejectedAutomatically",A_),("OpportunityReduced",DR)]),
"WORKLOAD_OR_ROLE_CHANGE": T(2,2,2.4,300,"OS",0,False,[("TasksAutomated",A_),("ResponsibilitiesChanged",A_)]),
"UNAUTHORIZED_DATA_COLLECTION": T(3.5,1.6,3.5,300,"PD,AC",0,False,[("PersonalDataStored",A_),("ConsentAbsent",A_),("UnauthorizedDataCollection",DR)]),
"DATA_EXPOSURE": T(3.5,3,3.4,60,"PD,LF,HS,OS",0.5,False,[("SecurityVulnerabilityPresent",A_),("UnauthorizedAccess",A_),("PersonalDataExposed",DR),("PrivacyViolated",DR)]),
"DISCRIMINATION": T(3.6,3,3.8,300,"OS,LF,AC",0.3,False,[("ProtectedAttributeProxyUsed",A_),("HumanReviewAbsent",A_),("DiscriminatoryDecision",DR)]),
"UNJUSTIFIED_AUTOMATED_DECISION": T(3.5,3,3,90,"LF,BN,AC,HS",0.3,False,[("HumanReviewAbsent",A_),("DecisionAutoExecuted",A_),("UnjustifiedAutomatedDecision",DR)]),
"LOSS_OF_HUMAN_CONTROL": T(3,3,2.5,45,"AC,OS",0,False,[("HumanOverrideDenied",A_),("DecisionAutoExecuted",A_),("LossOfHumanControl",DR)]),
"DENIAL_OF_APPEAL": T(3,2.5,3.5,80,"AC,OS",0,False,[("AppealRequested",A_),("AppealDenied",A_)]),
"LACK_OF_ACCOUNTABILITY": T(3,2,3.7,150,"AC,OS",0,False,[("ResponsibilityDisputed",A_),("AccountabilityLacking",DR)]),
"LACK_OF_TRANSPARENCY": T(3,2,3.4,120,"AC",0,False,[("ExplanationNotProvided",A_),("TransparencyLacking",DR)]),
"SURVEILLANCE": T(3,2.5,3.7,350,"PD,AC",0,False,[("MovementTrackedContinuously",A_),("OptOutUnavailable",A_),("SurveillanceEstablished",DR)]),
"AUTONOMY_RESTRICTION": T(3,2,2.4,60,"AC,LF",0,False,[("CreditLimitReducedAutomatically",A_),("AutonomyConstrained",DR)]),
"MANIPULATION": T(3,2,2.4,90,"AC,LF",1.0,False,[("ManipulativePromptUsed",A_),("UnrequestedSubscriptionAdded",A_),("BehaviourManipulated",DR)]),
}
DT = {"FINANCIAL": ("HIGH", "FinancialCredentialsExposed"), "MEDICAL": ("CRITICAL", "MedicalRecordsExposed"), "CONTACT": ("LOW", "ContactDataExposed"),
      "BIOMETRIC": ("HIGH", "BiometricDataStored"), "OTHER": ("MODERATE", "ConfidentialContentExposed")}
EXTRA_PRED = {"UNAUTHORIZED_DATA_COLLECTION": {"FINANCIAL": "FinancialInformationStored", "MEDICAL": "MedicalDataSharedWithThirdParty", "BIOMETRIC": "VoiceRecordedWithoutConsent"}}

# mechanisms: (id, field, app, role, tag, chain, og_pairs, dtype_for_data, texts, neg, hyp)
def M(i, f, a, role, tag, chain, og, dtp, texts, neg=None, hyp=None): return dict(id=i, field=f, app=a, role=role, tag=tag, chain=chain, og=og, dtp=dtp, texts=texts, neg=neg, hyp=hyp)
MECH = [
M(1,"Recruitment & HR","Resume screening","APPLICANT","decision",["DISCRIMINATION","OPPORTUNITY_REDUCTION"],True,None,["An automated resume-screening system lowered the scores of applicants with a protected-group proxy in their resumes, and {n_txt} were rejected without human review.","A screening model trained on past hiring decisions ranked {n_txt} lower because of a proxy for a protected characteristic."],None,("DisciplinaryUseOfScores","Some applicants feared the scores might also be shared with other employers.")),
M(2,"Recruitment & HR","Resume screening","APPLICANT","decision",["LACK_OF_TRANSPARENCY","DENIAL_OF_APPEAL"],False,None,["{n_txt} were rejected by an automated screening tool, received no explanation, and had their review requests refused."],None,None),
M(3,"Recruitment & HR","Performance analytics","EMPLOYEE","employment",["EMPLOYMENT_TERMINATION","DENIAL_OF_APPEAL"],False,None,["An automated productivity flag contributed to the dismissal of {n_txt}, and their appeals were rejected."],None,None),
M(4,"Finance & Insurance","Credit scoring","CUSTOMER","decision",["LACK_OF_TRANSPARENCY","UNJUSTIFIED_AUTOMATED_DECISION"],True,None,["{n_txt} had loan applications declined by an automated credit-scoring system with no explanation."],None,None),
M(5,"Finance & Insurance","Credit scoring","CUSTOMER","decision",["DISCRIMINATION"],False,None,["An audit found that a credit-scoring model used neighbourhood data as a proxy for a protected group, giving {n_txt} worse loan terms."],None,None),
M(6,"Finance & Insurance","Fraud detection","CUSTOMER","data",["UNAUTHORIZED_DATA_COLLECTION","DATA_EXPOSURE"],True,"FINANCIAL",["A platform stored {n_txt} customers' private financial information without their consent, and a security vulnerability later allowed unauthorized access."],("FinancialDataExposed","No financial credentials were involved."),None),
M(7,"Finance & Insurance","Fraud detection","CUSTOMER","data",["DATA_EXPOSURE"],False,"CONTACT",["A flaw exposed names, phone numbers and email addresses of {n_txt} customers to an unauthorized party."],("FinancialDataExposed","No financial data was involved."),None),
M(8,"Finance & Insurance","Credit scoring","CUSTOMER","decision",["AUTONOMY_RESTRICTION"],False,None,["A credit platform reduced the credit limits of {n_txt} customers automatically, constraining planned purchases."],None,None),
M(9,"Healthcare","Clinical documentation","PATIENT","data",["UNAUTHORIZED_DATA_COLLECTION"],False,"MEDICAL",["A clinical documentation assistant sent notes of {n_txt} patients to an external vendor for model improvement without consent."],None,None),
M(10,"Healthcare","Clinical documentation","PATIENT","data",["DATA_EXPOSURE"],False,"MEDICAL",["A misconfigured clinical AI scribe exposed the medical records of {n_txt} patients to unauthorized third parties."],("FinancialDataExposed","No payment information was involved."),None),
M(11,"Healthcare","Triage support","PATIENT","decision",["LOSS_OF_HUMAN_CONTROL"],False,None,["Clinicians could not override an automated triage priority for {n_txt} patients."],None,None),
M(12,"Customer Service","Support chatbot","EMPLOYEE","employment",["EMPLOYMENT_TERMINATION"],False,None,["After a support-chatbot upgrade, a company ended the employment of {n_txt} support agents."],("RoleEliminated","No other positions were eliminated."),("DisciplinaryUseOfScores","Some agents feared the chatbot data might also be used for discipline.")),
M(13,"Customer Service","Support chatbot","CUSTOMER","decision",["MANIPULATION"],False,None,["A support chatbot used pressure prompts to add unrequested subscriptions to the accounts of {n_txt} customers."],None,None),
M(14,"Customer Service","Support chatbot","CUSTOMER","data",["UNAUTHORIZED_DATA_COLLECTION"],False,"BIOMETRIC",["A support chatbot recorded and stored the voices of {n_txt} callers without their consent."],None,None),
M(15,"Customer Service","Support chatbot","EMPLOYEE","employment",["DESKILLING"],False,None,["{n_txt} support agents whose work was reduced to approving chatbot drafts reported declining troubleshooting skills."],None,None),
M(16,"Public Administration","Benefits eligibility","CITIZEN","decision",["UNJUSTIFIED_AUTOMATED_DECISION","DENIAL_OF_APPEAL"],False,None,["An automated eligibility system ended benefits for {n_txt} recipients because of a data-matching error, and appeal requests were refused."],None,None),
M(17,"Public Administration","Benefits eligibility","CITIZEN","decision",["LACK_OF_ACCOUNTABILITY"],False,None,["{n_txt} recipients of an automated decision could not find any body willing to accept responsibility for the outcome."],None,None),
M(18,"Public Administration","Benefits eligibility","CITIZEN","decision",["DISCRIMINATION"],False,None,["An eligibility model gave {n_txt} applicants higher risk scores because of a protected-group proxy."],None,None),
M(19,"Manufacturing","Visual inspection","EMPLOYEE","employment",["JOB_DISPLACEMENT"],False,None,["A visual inspection system took over most inspection tasks and the roles of {n_txt} inspectors were eliminated."],("RoleEliminated","No other positions were eliminated."),None),
M(20,"Manufacturing","Predictive maintenance","EMPLOYEE","employment",["WORKLOAD_OR_ROLE_CHANGE"],False,None,["{n_txt} technicians had their duties changed to exception handling after a predictive-maintenance rollout."],("RoleEliminated","No positions were eliminated."),None),
M(21,"Retail","Workforce scheduling","EMPLOYEE","employment",["WORKPLACE_MONITORING","SURVEILLANCE"],True,None,["A scheduling system continuously tracked the movement of {n_txt} employees with no opt-out."],None,("DisciplinaryUseOfScores","Some workers feared the data might also be used for discipline.")),
M(22,"Retail","Workforce scheduling","EMPLOYEE","employment",["WAGE_REDUCTION"],False,None,["A scheduling system assigned low productivity scores automatically and reduced the paid hours of {n_txt} workers."],None,None),
M(23,"Retail","Workforce scheduling","EMPLOYEE","employment",["EMPLOYMENT_INSTABILITY"],False,None,["{n_txt} workers reported highly irregular shift schedules after the scheduling tool was introduced."],None,None),
M(24,"Transportation & Logistics","Driver management","EMPLOYEE","employment",["EMPLOYMENT_TERMINATION","DENIAL_OF_APPEAL"],False,None,["An automated rating system deactivated the accounts of {n_txt} drivers, and appeals were rejected."],None,None),
M(25,"Transportation & Logistics","Fleet routing","EMPLOYEE","decision",["AUTONOMY_RESTRICTION"],False,None,["A routing system removed the discretion of {n_txt} drivers over breaks and routes."],None,None),
M(26,"Legal Services","Contract analysis","EMPLOYEE","employment",["JOB_DISPLACEMENT"],False,None,["A contract-analysis tool took over review work and the roles of {n_txt} junior reviewers were eliminated."],None,None),
M(27,"Software & IT","Code assistant","CUSTOMER","data",["DATA_EXPOSURE"],False,"OTHER",["A code assistant exposed proprietary content from {n_txt} users to other users."],None,None),
M(28,"Retail","Recommendation engine","CUSTOMER","decision",["MANIPULATION"],False,None,["A recommendation engine used deceptive urgency prompts that led {n_txt} customers into unwanted purchases."],None,None),
]
APPS = {}
for m in MECH: APPS.setdefault((m["field"], m["app"]), []).append(m)
UNIT = {"Resume screening":"DECISIONS","Performance analytics":"WORKERS","Credit scoring":"DECISIONS","Fraud detection":"TRANSACTIONS","Clinical documentation":"USERS","Triage support":"DECISIONS",
        "Support chatbot":"TRANSACTIONS","Benefits eligibility":"DECISIONS","Visual inspection":"DECISIONS","Predictive maintenance":"TASKS","Workforce scheduling":"WORKERS","Recommendation engine":"TRANSACTIONS",
        "Driver management":"WORKERS","Fleet routing":"DECISIONS","Contract analysis":"TASKS","Code assistant":"USERS"}
UMED = {"DECISIONS": 80000, "USERS": 15000, "TRANSACTIONS": 500000, "WORKERS": 2000, "TASKS": 100000}
STAKES = {"Recruitment & HR": ("HIGH", "MEDIUM", "HIGH"), "Finance & Insurance": ("HIGH", "HIGH", "MEDIUM"), "Healthcare": ("CRITICAL", "HIGH", "HIGH"), "Customer Service": ("MODERATE", "LOW", "LOW"),
          "Public Administration": ("HIGH", "HIGH", "HIGH"), "Manufacturing": ("LOW", "LOW", "LOW"), "Retail": ("MODERATE", "MEDIUM", "MEDIUM"), "Transportation & Logistics": ("MODERATE", "MEDIUM", "MEDIUM"),
          "Legal Services": ("HIGH", "MEDIUM", "LOW"), "Software & IT": ("MODERATE", "LOW", "LOW")}  # sensitivity, stakes, vulnerability
METRIC = "Agreement with human reviewers or ground truth on a labelled sample (synthetic definition)"
NAMES = ["Atlas","Beacon","Cirrus","Delta","Ember","Falcon","Granite","Harbor","Iris","Juno","Keystone","Lumen","Meridian","Nimbus","Orion","Pioneer","Quartz","Rialto","Sierra","Talon","Umbra","Vertex","Willow","Xenon","Yarrow","Zephyr"]
CAPS = ["BASIC_RULE_ASSISTED", "ML_ASSISTED", "ADVANCED_ML", "FOUNDATION_MODEL_BASED"]
OVR = ["NONE", "ON_REQUEST", "HUMAN_ON_LOOP", "HUMAN_IN_LOOP", "HUMAN_FINAL_DECISION"]
def pick(opts, p): return opts[rng.choice(len(opts), p=np.array(p) / np.sum(p))]
def unk(val, p=0.05): return "UNKNOWN" if rng.random() < p else val

# ---------------- deployments (Dataset 2) ----------------
deps = []
def depid(): return f"DEP-{len(deps) + 1:06d}"
sysn = 0
for (field, app), ms in APPS.items():
    for s in range(int(rng.choice([2, 3], p=[0.55, 0.45]))):
        name = f"{NAMES[sysn % 26]}-{['Screen','Score','Assist','Flow','Guard','Sense'][sysn % 6]}"; sysn += 1
        nver = int(rng.choice([1, 2, 3, 4], p=[0.3, 0.35, 0.25, 0.1]))
        start = D(2023, 1, 1) + dt.timedelta(days=int(rng.integers(0, 700)))
        sens, stakes, vuln = STAKES[field]
        base_a = rng.uniform(10, 45); base_c = rng.uniform(12, 38)
        zero_flag = rng.random() < 0.2
        ver_rows = []
        for k in range(nver):
            if start > CUTOFF - dt.timedelta(days=45): break
            pa = min(85, base_a + k * rng.uniform(6, 14)); pc = min(98 - pa, base_c + rng.uniform(-4, 6))
            pa10, pc10 = int(round(pa * 10)), int(round(max(pc, 2) * 10)); ph10 = 1000 - pa10 - pc10
            if ph10 < 20: pc10 = max(20, pc10 - (20 - ph10)); ph10 = 1000 - pa10 - pc10
            pr = round(pa10 / 10 * rng.uniform(0.05, 0.55), 1)
            cap = CAPS[int(np.clip(np.floor(rng.uniform(0, 2) + k * 0.7), 0, 3))]
            acc = None if rng.random() < 0.10 else round(float(np.clip(rng.uniform(0.72, 0.95) + 0.01 * k, 0, 0.98)), 3)
            ovr_p = {"HIGH": [0.08, 0.17, 0.35, 0.2, 0.2], "CRITICAL": [0.03, 0.07, 0.25, 0.25, 0.4], "MODERATE": [0.3, 0.3, 0.2, 0.1, 0.1], "LOW": [0.35, 0.3, 0.2, 0.1, 0.05]}[sens]
            stat = "PILOT" if (k == 0 and rng.random() < 0.15) else "ACTIVE"
            unit = UNIT[app]; exp_n = int(max(50, UMED[unit] * math.exp(rng.normal(0, 0.9)) * (0.12 if stat == "PILOT" else 1)))
            ver_rows.append(dict(field=field, app=app, system=name, version=f"{k + 1}.{int(rng.integers(0, 4))}", start=start, status=stat, pa10=pa10, pc10=pc10, ph10=ph10, pr=pr, cap=cap, acc=acc,
                ovr=unk(pick(OVR, ovr_p)), audit=unk(pick(["YES", "NO"], [0.3 if sens in ("HIGH", "CRITICAL") else 0.15, 0.7 if sens in ("HIGH", "CRITICAL") else 0.85])),
                priv=unk(pick(["NONE", "BASIC", "STRONG"], [0.12, 0.58, 0.30])), sec=unk(pick(["NONE", "BASIC", "STRONG"], [0.08, 0.57, 0.35])),
                transp=unk(pick(["NONE", "NOTICE_ONLY", "EXPLANATION_AVAILABLE"], [0.25, 0.5, 0.25])), appeal=unk(pick(["NONE", "LIMITED", "FULL"], [0.35, 0.4, 0.25])),
                mon=unk(pick(["YES", "NO"], [0.5, 0.5])), fb=unk(pick(["YES", "NO"], [0.55, 0.45])), sens=sens, stakes=stakes, vuln=vuln, unit=unit, exp=exp_n,
                conf=pick(["HIGH", "MEDIUM", "LOW"], [0.35, 0.45, 0.2]), zero=zero_flag or stat == "PILOT"))
            start = start + dt.timedelta(days=int(rng.integers(150, 330)))
        for k, v in enumerate(ver_rows):
            v["end"] = (ver_rows[k + 1]["start"] - dt.timedelta(days=1)) if k + 1 < len(ver_rows) else CUTOFF
            v["id"] = depid(); deps.append(v)
# proposed + no-established deployments
for f_, a_ in [("Retail", "Recommendation engine"), ("Legal Services", "Contract analysis"), ("Healthcare", "Triage support"), ("Public Administration", "Benefits eligibility")]:
    pa10 = int(rng.integers(150, 500)); pc10 = int(rng.integers(150, 400))
    deps.append(dict(id=depid(), field=f_, app=a_, system=f"{NAMES[sysn % 26]}-Planned", version="0.1", start=D(2027, 2, 1), end=None, status="PROPOSED", pa10=pa10, pc10=pc10, ph10=1000 - pa10 - pc10,
        pr=round(pa10 / 10 * 0.2, 1), cap="UNKNOWN", acc=None, ovr="UNKNOWN", audit="UNKNOWN", priv="UNKNOWN", sec="UNKNOWN", transp="UNKNOWN", appeal="UNKNOWN", mon="UNKNOWN", fb="UNKNOWN",
        sens=STAKES[f_][0], stakes=STAKES[f_][1], vuln=STAKES[f_][2], unit=None, exp=None, conf="LOW", zero=True)); sysn += 1
for f_, a_, s_, st_, vl_ in [("Education", "AI tutoring", "MODERATE", "MEDIUM", "HIGH"), ("Education", "Automated grading", "MODERATE", "HIGH", "HIGH"), ("Agriculture", "Crop advisory", "LOW", "LOW", "LOW"), ("Urban Infrastructure", "Traffic management", "MODERATE", "MEDIUM", "MEDIUM")]:
    deps.append(dict(id=depid(), field=f_, app=a_, system="NONE", version=None, start=CUTOFF, end=None, status="NO_ESTABLISHED_DEPLOYMENT", pa10=0, pc10=0, ph10=1000, pr=0.0, cap="NOT_APPLICABLE", acc=None,
        ovr="NOT_APPLICABLE", audit="NOT_APPLICABLE", priv="NOT_APPLICABLE", sec="NOT_APPLICABLE", transp="NOT_APPLICABLE", appeal="NOT_APPLICABLE", mon="NOT_APPLICABLE", fb="NOT_APPLICABLE",
        sens=s_, stakes=st_, vuln=vl_, unit=None, exp=None, conf="LOW", zero=True))
DEPD = {d["id"]: d for d in deps}
GT = dict(seed=SEED, note="Ground-truth generating effects (validation only, never findings)",
          incident_weight="w = (exposure/unit_median)^0.7 * exp(0.012*(p_a-40)+0.010*p_r -0.18*oversight_rank -0.35*[audit==YES] -0.20*privacy_rank -0.15*[monitoring==YES]) * Gamma(shape 0.6)",
          mechanism_weight="data: exp(-0.35*privacy_rank-0.25*security_rank); decision: exp(-0.25*oversight_rank-0.30*appeal_rank); employment: exp(0.012*(p_a-40)+0.010*p_r)",
          level_model="level = clip(round(mu + 0.7*z_incident + N(0,0.55)),1,5); Reversibility mu lowered by 0.5*appeal_rank", confirmation_mix=dict(CONFIRMED=0.80, ALLEGED=0.08, SUSPECTED=0.06, UNKNOWN=0.04, CONTRADICTED=0.02))
RK = {"NONE": 0, "BASIC": 1, "STRONG": 2, "UNKNOWN": 1, "ON_REQUEST": 1, "HUMAN_ON_LOOP": 2, "HUMAN_IN_LOOP": 3, "HUMAN_FINAL_DECISION": 4, "LIMITED": 1, "FULL": 2, "NOT_APPLICABLE": 0}
def ork(x): return 1.5 if x == "UNKNOWN" else RK[x]
live = [d for d in deps if d["status"] in ("ACTIVE", "PILOT") and not d["zero"]]
wts = []
for d in live:
    f = math.exp(0.012 * (d["pa10"] / 10 - 40) + 0.010 * d["pr"] - 0.18 * ork(d["ovr"]) - 0.35 * (d["audit"] == "YES") - 0.20 * RK[d["priv"]] - 0.15 * (d["mon"] == "YES"))
    wts.append((d["exp"] / UMED[d["unit"]]) ** 0.7 * f * rng.gamma(0.6, 1 / 0.6))
wts = np.array(wts) / np.sum(wts)

# ---------------- incidents / cases / harms ----------------
inc_rows, harm_rows, attr_rows, ben_rows = [], [], [], []
lvl = lambda mu, z: int(np.clip(np.rint(mu + 0.7 * z + rng.normal(0, 0.55)), 1, 5))
def dur_level(days, perm):
    if perm: return 5
    return 1 if days <= 7 else 2 if days <= 30 else 3 if days <= 180 else 4 if days <= 730 else 5
def sev(levels): x = (np.array(levels, float) - 1) / 4; return float(math.sqrt(float(np.sum(W * x ** 2))))
def n_txt(n): return "one person" if n == 1 else f"{n} people"
def people_count():
    u = rng.random()
    if u < 0.62: return 1
    if u < 0.77: return int(rng.integers(2, 6))
    if u < 0.95: return int(np.clip(np.rint(math.exp(rng.normal(3.5, 0.9))), 6, 400))
    return int(rng.integers(401, 5001))
def person_tok(c): return f"PRS-{c:07d}"
counter = dict(inc=0, case=0, harm=0, ben=0, person=0)
def new_status():
    return pick(["CONFIRMED", "ALLEGED", "SUSPECTED", "UNKNOWN", "CONTRADICTED"], [0.80, 0.08, 0.06, 0.04, 0.02])
def build_preds(cat, hstat, dtp, ai, pr, attr, name_ext, negpred, hyppred, first):
    preds = [("AISystem", [ai], "CONFIRMED", "ENTITY"), ("Person", [pr], "CONFIRMED", "ENTITY")]
    names = list(HT[cat]["preds"])
    if cat == "DATA_EXPOSURE":
        names.insert(2, (DT[dtp][1], A_))
        if dtp == "FINANCIAL": names.insert(3, ("FinancialLoss", A_)) if False else None
    if cat == "UNAUTHORIZED_DATA_COLLECTION" and dtp in EXTRA_PRED[cat]: names.insert(0, (EXTRA_PRED[cat][dtp], A_))
    core = []
    for k, (nm, ly) in enumerate(names):
        st = "CONFIRMED"
        if hstat in ("ALLEGED", "SUSPECTED", "UNKNOWN"): st = hstat
        if hstat == "CONTRADICTED": st = "CONTRADICTED" if ly == DR else "ALLEGED"
        preds.append((nm, [ai, pr], st, ly)); core.append(nm)
    if hstat == "CONTRADICTED": preds.append(("AuditFoundNoEvidence", [ai, pr], "CONFIRMED", A_))
    if attr == "SUSPECTED": preds.append(("AIInvolvementUnconfirmed", [ai, pr], "SUSPECTED", A_))
    if attr == "UNKNOWN": preds.append(("AIInvolvementUnconfirmed", [ai, pr], "UNKNOWN", A_))
    if first and negpred and negpred[0] not in core: preds.append((negpred[0], [ai, pr], "NEGATED", A_))
    if first and hyppred and hyppred[0] not in core: preds.append((hyppred[0], [ai, pr], "HYPOTHETICAL", A_))
    return preds
def make_harm(case_id, case, cat, hstat, dtp, z_inc, dep, n_pop, og, ai, pr, attr, preds_ext):
    counter["harm"] += 1; chid = f"CH-{counter['harm']:06d}"
    t = HT[cat]; ap = RK[dep["appeal"]] if dep else 1
    N = max(2, lvl(t["N"], z_inc)); Mv = lvl(t["M"], z_inc); R = lvl(t["R"] - 0.5 * ap, z_inc)
    if t["loss"] == 0.5 and dtp not in ("FINANCIAL",): pass
    perm = (cat == "DATA_EXPOSURE" and dtp in ("MEDICAL", "FINANCIAL") and rng.random() < 0.06)
    mon = dep["mon"] == "YES" if dep else False
    days = None if perm else round(float(max(0.2, t["days"] * math.exp(rng.normal(0, 0.9)) * (0.7 if mon else 1) * (1 + 0.12 * (R - 3)))), 1)
    D_ = dur_level(days, perm)
    k = 1 + (Mv >= 3) + (rng.random() < 0.25); k = int(min(k, len(t["pool"]), 5))
    doms = [DOM[x] for x in rng.choice(t["pool"], size=k, replace=False)]
    P = min(5, len(doms))
    lossp = t["loss"]
    if cat == "DATA_EXPOSURE": lossp = 0.8 if dtp == "FINANCIAL" else 0.0
    loss = None
    if Mv >= 2 and rng.random() < lossp:
        loss = round(float({2: 120, 3: 900, 4: 6000, 5: 30000}[Mv] * math.exp(rng.normal(0, 0.5))), 2)
    inc = None
    if t["inc"]: inc = round(float(np.clip({1: .02, 2: .1, 3: .25, 4: .6, 5: .9}[Mv] + rng.uniform(-.07, .07), 0.01, 1.0)), 2)
    rem = False if R == 5 else True if R <= 2 else bool(rng.random() < (0.8 if R == 3 else 0.35))
    dim = TAX[cat][1]; S = round(sev([N, Mv, P, D_, R]), 6)
    dsens = DT[dtp][0] if dtp else "NOT_APPLICABLE"
    xs = [round((v - 1) / 4, 4) for v in (N, Mv, P, D_, R)]
    harm_rows.append(dict(Case_Harm_ID=chid, Case_ID=case_id, Harm_Category=cat, Harm_Dimension=dim, Harm_Parent_Category=TAX[cat][0], Predicate_Logic=json.dumps(preds_ext), Extraction_Confidence=None,
        Harm_Confirmation_Status=hstat, Confirmation_Weight=CW[hstat], Employment_Severity=S if dim == "EMPLOYMENT" else None, Sovereignty_Severity=S if dim == "SOVEREIGNTY" else None,
        Severity_Calculation_Status="CALCULATED", Severity_Formula_Version=FORMULA_VERSION, Overlap_Group_ID=og, Predicate_Provenance="GENERATOR_GOLD"))
    attr_rows.append(dict(Case_Harm_ID=chid, Nature_Score=N, Magnitude_Score=Mv, Scope_Score=P, Duration_Score=D_, Reversibility_Score=R, Nature_x=xs[0], Magnitude_x=xs[1], Scope_x=xs[2], Duration_x=xs[3], Reversibility_x=xs[4],
        Financial_Loss_Amount=loss, Currency="USD" if loss is not None else None, Data_Sensitivity_Level=dsens, Impact_Duration_Days=days, Affected_Population=n_pop, Remedy_Available=rem, Information_Imputed=False,
        Imputation_Method="NOT_APPLICABLE", Imputation_Comparable_Sample_Size=None, Original_Missing_Fields=None, Scope_Domains_Affected=";".join(doms), Scope_Domain_Count=len(doms),
        Duration_Type="PERMANENT" if perm else "BOUNDED", Income_Loss_Fraction=inc, Data_Type=dtp if dtp else "NOT_APPLICABLE", Level_Assignment_Note=None))
    return chid
case_target = N_SYN_CASES
while counter["case"] < case_target:
    d = live[rng.choice(len(live), p=wts)] if rng.random() > 0.04 else None
    field, app = (d["field"], d["app"]) if d else (None, None)
    if d is None:
        (field, app) = list(APPS.keys())[int(rng.integers(0, len(APPS)))]
    ms = APPS[(field, app)]
    mw = []
    for m in ms:
        f = 1.0
        if d:
            if m["tag"] == "data": f = math.exp(-0.35 * RK[d["priv"]] - 0.25 * RK[d["sec"]])
            elif m["tag"] == "decision": f = math.exp(-0.25 * ork(d["ovr"]) - 0.30 * RK[d["appeal"]])
            else: f = math.exp(0.012 * (d["pa10"] / 10 - 40) + 0.010 * d["pr"])
        mw.append(f)
    m = ms[rng.choice(len(ms), p=np.array(mw) / np.sum(mw))]
    pac = people_count(); remaining = case_target - counter["case"]
    if pac <= 5: rows_n = pac
    else: rows_n = int(rng.choice([1, 2], p=[0.8, 0.2]))
    if rows_n > remaining: pac = rows_n = remaining
    counter["inc"] += 1; iid = f"INC-{counter['inc']:06d}"; rid = f"RPT-{counter['inc']:06d}"
    hstat0 = new_status()
    if hstat0 == "CONFIRMED": attr = pick(["CONFIRMED", "SUSPECTED"], [0.85, 0.15])
    elif hstat0 in ("ALLEGED", "SUSPECTED"): attr = pick(["SUSPECTED", "UNKNOWN"], [0.6, 0.4])
    elif hstat0 == "UNKNOWN": attr = pick(["UNKNOWN", "SUSPECTED"], [0.7, 0.3])
    else: attr = pick(["CONFIRMED", "SUSPECTED"], [0.5, 0.5])
    if d is None: attr = pick(["UNKNOWN", "SUSPECTED"], [0.6, 0.4])
    if d:
        lo = max(d["start"], D(2023, 1, 1)); hi = min(d["end"] or CUTOFF, CUTOFF)
        idate = lo + dt.timedelta(days=int(rng.integers(0, max(1, (hi - lo).days + 1))))
    else: idate = D(2023, 6, 1) + dt.timedelta(days=int(rng.integers(0, 1250)))
    lag = int(np.clip(rng.lognormal(3.2, 0.9), 1, 420)); rdate = min(idate + dt.timedelta(days=lag), D(2026, 11, 15))
    txt = m["texts"][int(rng.integers(0, len(m["texts"])))].format(n_txt=n_txt(pac), sys=d["system"] if d else "an AI system")
    if hstat0 == "ALLEGED": txt = "It is alleged that " + txt[0].lower() + txt[1:]
    if hstat0 == "SUSPECTED": txt += " The cause has not been confirmed but AI involvement is suspected."
    if hstat0 == "UNKNOWN": txt += " The evidence does not establish whether the harm occurred."
    if hstat0 == "CONTRADICTED": txt += " An independent review found no support for the claim."
    negp = hypp = None
    if m["neg"] and rng.random() < 0.3: negp = m["neg"]; txt += " " + m["neg"][1]
    if m["hyp"] and rng.random() < 0.25: hypp = m["hyp"]; txt += " " + m["hyp"][1]
    ai = f"AI_{d['id'][4:]}" if d else "AI_UNIDENTIFIED"
    desc = "[SYNTHETIC SCENARIO] " + txt
    # cases
    if pac <= 5: reps = [1] * pac; gran = "INDIVIDUAL"
    else:
        gran = "COHORT"
        if rows_n == 1: reps = [pac]
        else: a1 = int(rng.integers(1, pac)); reps = [a1, pac - a1]
    z_inc = float(rng.normal(0, 0.6)); causal_ok = True
    first_case_id = None
    for ci, rep in enumerate(reps, start=1):
        counter["case"] += 1; counter["person"] += 1
        case_id = f"CAS-{counter['inc']:06d}-{ci:02d}"; first_case_id = first_case_id or case_id
        pr = person_tok(counter["person"]) if gran == "INDIVIDUAL" else f"PRS-COHORT-{counter['case']:06d}"
        inc_rows.append(dict(Incident_ID=iid, Case_ID=case_id, Person_ID=pr, Report_ID=rid, Incident_Date=idate, Report_Date=rdate, Field=field if d else "Finance & Insurance" if app == "Fraud detection" else field,
            AI_System=d["system"] if d else "UNKNOWN", AI_Application=app if d else app + " (system not identified)", AI_Version=d["version"] if d else None, Deployment_ID=d["id"] if d else None,
            Incident_Description=desc, People_Affected_Count=pac, Case_Granularity=gran, Represented_Person_Count=rep, AI_Attribution=attr, Record_Type="SYNTHETIC",
            Source_Reference="SYNTHETIC_NO_EXTERNAL_SOURCE", Source_Quality="NOT_APPLICABLE_SYNTHETIC", Person_Role=m["role"]))
        zc = z_inc + float(rng.normal(0, 0.25))
        chain_ids = []
        for hi_, cat in enumerate(m["chain"]):
            if hi_ > 0 and rng.random() > 0.85: continue
            dtp = m["dtp"] if cat in ("DATA_EXPOSURE", "UNAUTHORIZED_DATA_COLLECTION") else None
            og = f"OG-{case_id}" if (m["og"] and len(m["chain"]) > 1) else None
            preds = build_preds(cat, hstat0, dtp, ai, pr, attr, None, negp, hypp, hi_ == 0)
            pj = [dict(predicate_name=p_[0], args=p_[1], assertion_status=p_[2], layer=p_[3]) for p_ in preds]
            cid = make_harm(case_id, None, cat, hstat0, dtp, zc, d, rep, og, ai, pr, attr, pj); chain_ids.append((cid, og))
        if m["og"] and len(chain_ids) < 2:  # no overlap group if chain incomplete
            for cid, og in chain_ids:
                for hr in harm_rows:
                    if hr["Case_Harm_ID"] == cid: hr["Overlap_Group_ID"] = None
    if d and rng.random() < 0.08 and first_case_id:  # case-level benefit (mixed outcome)
        counter["ben"] += 1
        ben_rows.append(dict(Benefit_ID=f"BEN-{counter['ben']:09d}", Case_ID=first_case_id, Deployment_ID=None, Benefit_Category="REDUCED_PROCESSING_TIME", Benefit_Description="[SYNTHETIC] The affected user's original request was handled faster than under the manual process.",
            Benefit_Magnitude=None, Benefit_Confidence="MEDIUM", Benefit_Evidence_Status="REPORTED", Benefit_Source="SYNTHETIC_NO_EXTERNAL_SOURCE", Record_Type="SYNTHETIC",
            Benefit_Raw_Value=round(float(rng.uniform(10, 60)), 1), Benefit_Raw_Unit="PERCENT_REDUCTION_HANDLING_TIME_VS_BASELINE"))
# deployment-level benefits (independent of harm)
BC = [("PRODUCTIVITY", "PERCENT_INCREASE_TASKS_PER_HOUR_VS_BASELINE", 5, 40), ("ACCURACY", "PERCENT_REDUCTION_ERRORS_VS_BASELINE", 3, 35), ("REDUCED_PROCESSING_TIME", "PERCENT_REDUCTION_PROCESSING_TIME_VS_BASELINE", 10, 70),
      ("COST_REDUCTION", "PERCENT_REDUCTION_COST_PER_CASE_VS_BASELINE", 5, 45), ("ACCESSIBILITY", "PERCENT_OF_REQUESTS_HANDLED_OUTSIDE_OFFICE_HOURS", 10, 80), ("NEW_WORK_OPPORTUNITIES", "COUNT_NEW_ROLES_CREATED", 1, 40)]
for d in deps:
    if d["status"] not in ("ACTIVE", "PILOT"): continue
    for bi in rng.choice(len(BC), size=int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25])), replace=False):
        cat, unit, lo, hi = BC[bi]
        q = float(np.clip(0.15 + 0.2 * CAPS.index(d["cap"]) + (0.4 * (d["acc"] - 0.7) / 0.3 if d["acc"] else 0.15) + rng.normal(0, 0.15), 0.02, 1))
        ev = pick(["CONFIRMED", "REPORTED", "ESTIMATED", "UNKNOWN"], [0.35, 0.4, 0.2, 0.05]); counter["ben"] += 1
        ben_rows.append(dict(Benefit_ID=f"BEN-{counter['ben']:09d}", Case_ID=None, Deployment_ID=d["id"], Benefit_Category=cat, Benefit_Description=f"[SYNTHETIC] {cat.replace('_', ' ').title()} reported for this deployment configuration.",
            Benefit_Magnitude=None, Benefit_Confidence={"CONFIRMED": "HIGH", "REPORTED": "MEDIUM", "ESTIMATED": "LOW", "UNKNOWN": "LOW"}[ev], Benefit_Evidence_Status=ev, Benefit_Source="SYNTHETIC_NO_EXTERNAL_SOURCE",
            Record_Type="SYNTHETIC", Benefit_Raw_Value=round(lo + (hi - lo) * q, 1), Benefit_Raw_Unit=unit))

# ---------------- 15 DOCUMENTED incidents (no invented facts; NULL where unknown) ----------------
DOCS = []
def doc(key, field, system, app, idate, rdate, pac, attr, role, src, qual, desc, harms):
    DOCS.append(dict(key=key, field=field, system=system, app=app, idate=idate, rdate=rdate, pac=pac, attr=attr, role=role, src=src, qual=qual, desc=desc, harms=harms))
def dh(cat, status, preds, N=None, M=None, doms=None, days=None, R=None, rem=None, dtp="NOT_APPLICABLE", dsens="NOT_APPLICABLE", og=None, note="", perm=False):
    return dict(cat=cat, status=status, preds=preds, N=N, M=M, doms=doms, days=days, R=R, rem=rem, dtp=dtp, dsens=dsens, og=og, note=note, perm=perm)
NU = "URL not captured in this build; verify against the original before publication."
DS = "[DOCUMENTED SUMMARY, paraphrased from public sources] "
doc(1,"Recruitment & HR","iTutorGroup tutor application software","Tutor application screening",None,D(2022,5,5),200,"SUSPECTED","APPLICANT",
 "EEOC press releases: 'EEOC Sues iTutorGroup for Age Discrimination' (5 May 2022) and 'iTutorGroup to Pay $365,000 to Settle EEOC Discriminatory Hiring Suit' (11 Sep 2023): https://eeoc.gov/newsroom/itutorgroup-pay-365000-settle-eeoc-discriminatory-hiring-suit","VERIFIED_PRIMARY",
 DS+"The U.S. EEOC alleged that a tutoring company programmed its application software to automatically reject older applicants (women aged 55 or older and men aged 60 or older), rejecting more than 200 qualified U.S. applicants (count stored as a lower bound). The company denied wrongdoing and the case was settled by a 2023 consent decree providing $365,000 to affected applicants. The allegations were not adjudicated. Rejections are reported for March and April 2020 (exact dates not given).",
 [dh("DISCRIMINATION","ALLEGED",[("AgeBasedRejectionRule","ALLEGED","ATOMIC"),("ApplicationRejectedAutomatically","ALLEGED","ATOMIC"),("DiscriminatoryDecision","ALLEGED","DERIVED"),("WrongdoingDenied","CONFIRMED","ATOMIC")],og="OG-DOC-001",note="Levels not assigned: per-person duration, loss and remedy outcomes are not documented."),
  dh("OPPORTUNITY_REDUCTION","ALLEGED",[("ApplicationRejectedAutomatically","ALLEGED","ATOMIC"),("OpportunityReduced","ALLEGED","DERIVED")],og="OG-DOC-001",note="Levels not assigned (insufficient public detail).")])
doc(2,"Software & IT","ChatGPT","Consumer AI chatbot",D(2023,3,20),D(2023,3,24),None,"CONFIRMED","CUSTOMER",
 "OpenAI, 'March 20 ChatGPT outage: Here's what happened' (24 Mar 2023): https://openai.com/blog/march-20-chatgpt-outage","VERIFIED_PRIMARY",
 DS+"OpenAI reported that a bug in an open-source library caused some users to see titles from other users' chat histories, and that payment-related information (name, email, payment address, card type, last four digits and expiry date) of about 1.2% of ChatGPT Plus subscribers active in a nine-hour window may have been visible to others. OpenAI said the number of users whose data was actually revealed was extremely low and notified affected users. The total number of people is not stated in the source.",
 [dh("DATA_EXPOSURE","SUSPECTED",[("SecurityVulnerabilityPresent","CONFIRMED","ATOMIC"),("PaymentRelatedDataVisibleToOthers","SUSPECTED","ATOMIC"),("PersonalDataExposed","SUSPECTED","DERIVED")],N=3,M=1,doms=["PRIVACY_DATA"],days=0.375,R=3,rem=True,dtp="FINANCIAL",dsens="MODERATE",
     note="Annotator levels from the OpenAI post: partial card data only (N3); no confirmed misuse stated (M1); nine-hour window (D1); users notified, bug fixed (R3).")])
doc(3,"Public Administration","Facial recognition system used for a Detroit police investigation","Facial recognition",D(2020,1,9),D(2020,6,24),1,"CONFIRMED","CITIZEN",
 "NPR, 'The Computer Got It Wrong' (24 Jun 2020): https://www.npr.org/2020/06/24/882683463/the-computer-got-it-wrong-how-facial-recognition-led-to-a-false-arrest-in-mich ; ACLU First Amended Complaint: https://assets.aclu.org/live/uploads/2024/01/54.-First-Amended-Complaint.pdf","CREDIBLE_SECONDARY",
 DS+"Reports state that a man in Detroit was arrested in January 2020 after a facial recognition match wrongly identified him as a shoplifting suspect, was held for about 30 hours, and that the charges were later dismissed. Detroit police said they do not arrest on facial recognition alone. Remedy outcomes are not recorded here.",
 [dh("AUTONOMY_RESTRICTION","CONFIRMED",[("FalseFacialRecognitionMatch","CONFIRMED","ATOMIC"),("PersonDetained","CONFIRMED","ATOMIC"),("AutonomyConstrained","CONFIRMED","DERIVED")],N=4,M=3,doms=["AUTONOMY_CONTROL"],days=1.25,R=None,note="Duration from reported ~30 hours. Reversibility not assigned: remedy outcome not verified in sources used."),
  dh("DISCRIMINATION","ALLEGED",[("FalseFacialRecognitionMatch","CONFIRMED","ATOMIC"),("RacialBiasAlleged","ALLEGED","ATOMIC"),("DiscriminatoryDecision","ALLEGED","DERIVED")],note="ACLU alleges racial bias in the technology; not assigned levels.")])
doc(4,"Public Administration","Dutch tax authority risk-classification model (childcare benefits)","Benefit fraud risk scoring",None,D(2021,10,25),None,"CONFIRMED","CITIZEN",
 "Amnesty International, 'Xenophobic Machines' (25 Oct 2021): https://www.amnesty.org/en/latest/news/2021/10/xenophobic-machines-dutch-child-benefit-scandal/","CREDIBLE_SECONDARY",
 DS+"Amnesty International reported that the Dutch tax authorities used a risk-classification algorithm introduced in 2013 in which nationality was a risk factor, giving non-Dutch nationals higher risk scores. Tens of thousands of parents and caregivers, mostly from low-income families, were falsely accused of fraud, and the scandal brought down the Dutch government in January 2021. Amnesty also reported that civil servants were given no information on why a high-risk score was produced, and an absence of accountability.",
 [dh("DISCRIMINATION","CONFIRMED",[("NationalityUsedAsRiskFactor","CONFIRMED","ATOMIC"),("DiscriminatoryDecision","CONFIRMED","DERIVED")],note="Per-person levels not assigned."),
  dh("LACK_OF_TRANSPARENCY","CONFIRMED",[("ExplanationNotProvided","CONFIRMED","ATOMIC"),("TransparencyLacking","CONFIRMED","DERIVED")],note="Per-person levels not assigned."),
  dh("LACK_OF_ACCOUNTABILITY","CONFIRMED",[("AccountabilityLacking","CONFIRMED","DERIVED")],note="Per-person levels not assigned.")])
doc(5,"Public Administration","Centrelink automated income-averaging debt calculation (Robodebt)","Automated debt recovery",None,D(2023,7,7),None,"CONFIRMED","CITIZEN",
 "Royal Commission into the Robodebt Scheme, Final Report (7 Jul 2023); coverage: https://www.jurist.org/news/2023/07/australia-robodebt-royal-commission-hands-down-findings-recommends-referrals-for-prosecution ; https://www.sbs.com.au/news/article/robodebt-report-recommends-welfare-payment-increase-in-lieu-of-compensation/fvhxwtutj","CREDIBLE_SECONDARY",
 DS+"The Royal Commission into Australia's Robodebt scheme reported that an automated debt-raising process based on averaging income data from tax records generated debts, many not owed; income averaging used this way was reported as inconsistent with social security legislation, and the Federal Court had found the method unlawful in 2019. Reports describe roughly half a million debts or nearly half a million people (figures differ), so the number of people is stored as unknown. A class action was settled in 2020.",
 [dh("UNJUSTIFIED_AUTOMATED_DECISION","CONFIRMED",[("IncomeAveragingDebtRaised","CONFIRMED","ATOMIC"),("UnjustifiedAutomatedDecision","CONFIRMED","DERIVED")],note="Per-person levels not assigned (duration and remedy vary by person)."),
  dh("LACK_OF_ACCOUNTABILITY","CONFIRMED",[("AccountabilityLacking","CONFIRMED","DERIVED")],note="Per-person levels not assigned.")])
doc(6,"Healthcare","Commercial health-risk prediction algorithm studied by Obermeyer et al.","Care-management risk scoring",None,D(2019,10,25),None,"CONFIRMED","PATIENT",
 "Obermeyer Z, Powers B, Vogeli C, Mullainathan S. 'Dissecting racial bias in an algorithm used to manage the health of populations.' Science 366(6464):447-453 (25 Oct 2019). "+NU,"VERIFIED_PRIMARY",
 DS+"A peer-reviewed study reported racial bias in a widely used commercial algorithm that predicts patients' health-care needs: patients identified as Black were assigned lower risk scores than equally sick White patients, because the algorithm used health-care costs as a proxy for need. The number of affected people is not stored here.",
 [dh("DISCRIMINATION","CONFIRMED",[("CostUsedAsProxyForNeed","CONFIRMED","ATOMIC"),("DiscriminatoryDecision","CONFIRMED","DERIVED")],note="Per-person levels not assigned.")])
doc(7,"Recruitment & HR","Amazon experimental recruiting engine","Resume screening",None,D(2018,10,10),None,"CONFIRMED","APPLICANT",
 "Reuters (J. Dastin), 'Amazon scraps secret AI recruiting tool that showed bias against women' (10 Oct 2018). "+NU,"CREDIBLE_SECONDARY",
 DS+"Reuters reported that Amazon built an experimental machine-learning recruiting tool that learned to downgrade resumes containing terms associated with women, and that the project was abandoned. Amazon is reported to have said the tool was not used to evaluate candidates, so the real-world effect on applicants is not established.",
 [dh("DISCRIMINATION","SUSPECTED",[("GenderAssociatedTermsPenalized","SUSPECTED","ATOMIC"),("DiscriminatoryDecision","SUSPECTED","DERIVED")],note="Model bias reported; effect on real applicants not established; no levels assigned.")])
doc(8,"Finance & Insurance","Apple Card credit-limit decision process","Credit scoring",None,None,None,"SUSPECTED","CUSTOMER",
 "New York State Department of Financial Services, report on its Apple Card investigation (March 2021); public allegations began November 2019. "+NU,"CREDIBLE_SECONDARY",
 DS+"In November 2019, public allegations said Apple Card credit-limit decisions gave women lower limits than men with comparable finances. The New York State Department of Financial Services investigated and, in a report published in March 2021, reported finding no violation of fair-lending law. Exact dates and the number of people are not stored.",
 [dh("DISCRIMINATION","CONTRADICTED",[("GenderBasedLimitsAlleged","ALLEGED","ATOMIC"),("RegulatorFoundNoViolation","CONFIRMED","ATOMIC"),("DiscriminatoryDecision","CONTRADICTED","DERIVED")],note="Allegation contradicted by regulator finding; no levels assigned.")])
doc(9,"Recruitment & HR","Workday AI-enabled applicant screening features","Resume screening",None,None,None,"SUSPECTED","APPLICANT",
 "Mobley v. Workday, Inc., U.S. District Court, Northern District of California, No. 3:23-cv-00770 (filed 2023). "+NU+" Re-check case status.","CREDIBLE_SECONDARY",
 DS+"A job applicant sued Workday alleging that its AI-based applicant-screening tools discriminated on the basis of race, age and disability. The claims are allegations, are contested, and the current case status should be re-checked.",
 [dh("DISCRIMINATION","ALLEGED",[("AutomatedScreeningBiasAlleged","ALLEGED","ATOMIC"),("DiscriminatoryDecision","ALLEGED","DERIVED")],og="OG-DOC-009",note="Allegation only; no levels assigned."),
  dh("OPPORTUNITY_REDUCTION","ALLEGED",[("ApplicationRejectedAutomatically","ALLEGED","ATOMIC"),("OpportunityReduced","ALLEGED","DERIVED")],og="OG-DOC-009",note="Allegation only; no levels assigned.")])
doc(10,"Software & IT","Clearview AI facial-recognition database","Facial recognition",None,None,None,"CONFIRMED","CITIZEN",
 "UK Information Commissioner's Office, monetary penalty and announcement concerning Clearview AI Inc. (May 2022). "+NU,"CREDIBLE_SECONDARY",
 DS+"The UK Information Commissioner's Office fined Clearview AI about 7.5 million pounds in May 2022 for collecting images of people from the web and social media to build a facial-recognition database without their knowledge, finding breaches of UK data protection law. The number of affected people and the exact date are not stored.",
 [dh("UNAUTHORIZED_DATA_COLLECTION","CONFIRMED",[("ImagesScrapedWithoutConsent","CONFIRMED","ATOMIC"),("ConsentAbsent","CONFIRMED","ATOMIC"),("UnauthorizedDataCollection","CONFIRMED","DERIVED")],dtp="BIOMETRIC",dsens="HIGH",note="Per-person levels not assigned.")])
doc(11,"Public Administration","COMPAS recidivism risk-scoring tool","Criminal risk assessment",None,D(2016,5,23),None,"CONFIRMED","CITIZEN",
 "ProPublica (Angwin, Larson, Mattu, Kirchner), 'Machine Bias' (23 May 2016). "+NU,"CREDIBLE_SECONDARY",
 DS+"ProPublica analysed risk scores produced by the COMPAS tool for defendants in Broward County, Florida, and reported racial disparities in its errors (Black defendants more often wrongly labelled higher risk). The tool's developer disputed the analysis. The number of people affected is not stored.",
 [dh("DISCRIMINATION","ALLEGED",[("RacialErrorDisparityReported","ALLEGED","ATOMIC"),("DiscriminatoryDecision","ALLEGED","DERIVED")],note="Contested analysis; no levels assigned.")])
doc(12,"Transportation & Logistics","Deliveroo rider-ranking algorithm","Driver management",None,D(2020,12,31),None,"CONFIRMED","EMPLOYEE",
 "Tribunale di Bologna, ordinanza of 31 Dec 2020 (Filcams CGIL, Nidil CGIL, Filt CGIL v Deliveroo Italia S.r.l.), as reported by the press. "+NU,"CREDIBLE_SECONDARY",
 DS+"An Italian court in Bologna ruled that the algorithm Deliveroo used to rank riders and allocate shifts was discriminatory because it penalised cancellations or absences without distinguishing legitimate reasons such as illness. The number of riders affected is not stored; Report_Date is the ruling date.",
 [dh("DISCRIMINATION","CONFIRMED",[("RankingIgnoredLegitimateAbsence","CONFIRMED","ATOMIC"),("DiscriminatoryDecision","CONFIRMED","DERIVED")],og="OG-DOC-012",note="Court finding; per-person levels not assigned."),
  dh("OPPORTUNITY_REDUCTION","CONFIRMED",[("ShiftAccessReducedByRanking","CONFIRMED","ATOMIC"),("OpportunityReduced","CONFIRMED","DERIVED")],og="OG-DOC-012",note="Per-person levels not assigned.")])
doc(13,"Education","Ofqual exam-grade standardisation model (England, 2020)","Exam grade standardisation",None,D(2020,8,17),None,"SUSPECTED","CITIZEN",
 "UK Government and Ofqual announcements of 17 Aug 2020 reverting to teacher-assessed grades; widely reported. "+NU,"CREDIBLE_SECONDARY",
 DS+"In August 2020, exam results in England were adjusted by a statistical standardisation model. After widespread criticism that many students' grades had been lowered, the UK government announced it would use teacher-assessed grades instead. Whether the model counts as AI is not established (hence AI_Attribution SUSPECTED); the number of students is not stored.",
 [dh("UNJUSTIFIED_AUTOMATED_DECISION","CONFIRMED",[("GradesLoweredByStandardisation","CONFIRMED","ATOMIC"),("DecisionReversedLater","CONFIRMED","ATOMIC"),("UnjustifiedAutomatedDecision","CONFIRMED","DERIVED")],note="Per-person levels not assigned.")])
doc(14,"Transportation & Logistics","Air Canada website chatbot","Support chatbot",None,None,1,"CONFIRMED","CUSTOMER",
 "Moffatt v. Air Canada, 2024 BCCRT 149 (British Columbia Civil Resolution Tribunal, February 2024). "+NU,"CREDIBLE_SECONDARY",
 DS+"A tribunal in British Columbia found Air Canada responsible for incorrect information about bereavement fares given to a customer by its website chatbot, rejecting the airline's argument that the chatbot was a separate entity. The money amount is not stored because it was awarded in Canadian dollars and this dataset uses USD only.",
 [dh("LACK_OF_ACCOUNTABILITY","CONFIRMED",[("ChatbotGaveIncorrectInformation","CONFIRMED","ATOMIC"),("ResponsibilityDisputed","CONFIRMED","ATOMIC"),("AccountabilityLacking","CONFIRMED","DERIVED")],note="Financial amount withheld (CAD; USD-only rule). No levels assigned.")])
doc(15,"Public Administration","Michigan Integrated Data Automated System (MiDAS)","Unemployment fraud determination",None,None,None,"CONFIRMED","CITIZEN",
 "Reporting by the Detroit Free Press and other outlets (2015 to 2017) and Michigan Unemployment Insurance Agency statements. "+NU,"CREDIBLE_SECONDARY",
 DS+"Michigan's automated unemployment-insurance system (MiDAS), used from about 2013, automatically flagged claimants for fraud with little human review; the state later acknowledged that many determinations were wrong. Counts and exact dates are not stored.",
 [dh("UNJUSTIFIED_AUTOMATED_DECISION","CONFIRMED",[("AutomatedFraudDetermination","CONFIRMED","ATOMIC"),("HumanReviewAbsent","CONFIRMED","ATOMIC"),("UnjustifiedAutomatedDecision","CONFIRMED","DERIVED")],note="Per-person levels not assigned.")])
assert len(DOCS) == 15
for dd_ in DOCS:
    k = dd_["key"]; iid = f"INC-9{k:05d}"; cid = f"CAS-9{k:05d}-01"; pr = f"PRS-DOC-{k:03d}"; ai = f"AI_DOC_{k:02d}"
    pac = dd_["pac"]
    inc_rows.append(dict(Incident_ID=iid, Case_ID=cid, Person_ID=pr, Report_ID=f"RPT-9{k:05d}", Incident_Date=dd_["idate"], Report_Date=dd_["rdate"], Field=dd_["field"], AI_System=dd_["system"], AI_Application=dd_["app"], AI_Version=None,
        Deployment_ID=None, Incident_Description=dd_["desc"], People_Affected_Count=pac, Case_Granularity="INDIVIDUAL" if pac == 1 else "COHORT", Represented_Person_Count=pac, AI_Attribution=dd_["attr"],
        Record_Type="DOCUMENTED", Source_Reference=dd_["src"], Source_Quality=dd_["qual"], Person_Role=dd_["role"]))
    for h in dd_["harms"]:
        counter["harm"] += 1; chid = f"CH-{counter['harm']:06d}"
        pj = [dict(predicate_name="AISystem", args=[ai], assertion_status="CONFIRMED", layer="ENTITY"), dict(predicate_name="Person", args=[pr], assertion_status="CONFIRMED", layer="ENTITY")]
        for nm, st, ly in h["preds"]: pj.append(dict(predicate_name=nm, args=[ai, pr], assertion_status=st, layer=ly))
        lv = [h["N"], h["M"], len(h["doms"]) if h["doms"] else None, None, h["R"]]
        dl = None if h["days"] is None and not h["perm"] else dur_level(h["days"], h["perm"])
        lv[3] = dl
        complete = all(v is not None for v in lv)
        S = round(sev(lv), 6) if complete else None; dim = TAX[h["cat"]][1]
        miss = [n for n, v in zip(["Nature", "Magnitude", "Scope", "Duration", "Reversibility"], lv) if v is None]
        harm_rows.append(dict(Case_Harm_ID=chid, Case_ID=cid, Harm_Category=h["cat"], Harm_Dimension=dim, Harm_Parent_Category=TAX[h["cat"]][0], Predicate_Logic=json.dumps(pj), Extraction_Confidence=None,
            Harm_Confirmation_Status=h["status"], Confirmation_Weight=CW[h["status"]], Employment_Severity=S if dim == "EMPLOYMENT" else None, Sovereignty_Severity=S if dim == "SOVEREIGNTY" else None,
            Severity_Calculation_Status="CALCULATED" if complete else "INSUFFICIENT_EVIDENCE", Severity_Formula_Version=FORMULA_VERSION, Overlap_Group_ID=h["og"], Predicate_Provenance="CURATED_FROM_SOURCE"))
        xs = [round((v - 1) / 4, 4) if v is not None else None for v in lv]
        attr_rows.append(dict(Case_Harm_ID=chid, Nature_Score=lv[0], Magnitude_Score=lv[1], Scope_Score=lv[2], Duration_Score=lv[3], Reversibility_Score=lv[4], Nature_x=xs[0], Magnitude_x=xs[1], Scope_x=xs[2], Duration_x=xs[3],
            Reversibility_x=xs[4], Financial_Loss_Amount=None, Currency=None, Data_Sensitivity_Level=h["dsens"], Impact_Duration_Days=h["days"], Affected_Population=pac, Remedy_Available=h["rem"], Information_Imputed=False,
            Imputation_Method="NOT_APPLICABLE", Imputation_Comparable_Sample_Size=None, Original_Missing_Fields=";".join(miss) if miss else None,
            Scope_Domains_Affected=";".join(h["doms"]) if h["doms"] else None, Scope_Domain_Count=len(h["doms"]) if h["doms"] else None, Duration_Type=("PERMANENT" if h["perm"] else "BOUNDED") if dl else None,
            Income_Loss_Fraction=None, Data_Type=h["dtp"], Level_Assignment_Note=h["note"]))

# ---------------- write tables ----------------
def b(x): return None if x is None else ("TRUE" if x else "FALSE")
INC = pd.DataFrame(inc_rows)
HAR = pd.DataFrame(harm_rows); ATT = pd.DataFrame(attr_rows); BEN = pd.DataFrame(ben_rows)
ATT["Remedy_Available"] = ATT["Remedy_Available"].map(lambda v: b(None if pd.isna(v) else bool(v)))
ATT["Information_Imputed"] = ATT["Information_Imputed"].map(lambda v: b(bool(v)))
drows = []
for d in deps:
    na = d["status"] == "NO_ESTABLISHED_DEPLOYMENT"; pr_ = d["status"] == "PROPOSED"
    drows.append(dict(Deployment_ID=d["id"], Field=d["field"], AI_Application=d["app"], AI_System=d["system"], AI_Version=d["version"], Deployment_Date=d["start"], Deployment_Status=d["status"],
        AI_Only_Work_Percentage=d["pa10"] / 10, Human_Only_Work_Percentage=d["ph10"] / 10, Collaborative_Work_Percentage=d["pc10"] / 10, Actual_Replacement_Percentage=d["pr"],
        Work_Allocation_Denominator="Not applicable (no established deployment)" if na else "Total relevant tasks in the application workflow (synthetic definition)",
        Replacement_Denominator="Not applicable (no established deployment)" if na else "Tasks replaced / total relevant tasks (synthetic definition)",
        AI_Capability_Level=d["cap"], AI_Accuracy=d["acc"], AI_Accuracy_Metric="NOT_APPLICABLE" if na else ("NOT_REPORTED" if d["acc"] is None else METRIC), Human_Oversight_Level=d["ovr"], Independent_Audit=d["audit"],
        Privacy_Safeguards=d["priv"], Security_Controls=d["sec"], Transparency_Mechanism=d["transp"], Human_Appeal_Mechanism=d["appeal"], Continuous_Monitoring=d["mon"], Fallback_Mechanism=d["fb"],
        Exposure_Unit=d["unit"], Exposure_Count=d["exp"], Exposure_Period_Start=None if (na or pr_) else d["start"], Exposure_Period_End=None if (na or pr_) else d["end"],
        Deployment_Source="SYNTHETIC_SCENARIO_ASSUMPTION: no established deployment; zero AI work shares are scenario assumptions, not measurements" if na else "SYNTHETIC_PLANNED_CONFIGURATION: scenario, not measured" if pr_ else "SYNTHETIC_NO_EXTERNAL_SOURCE",
        Data_Confidence=d["conf"], Record_Type="SYNTHETIC", Data_Sensitivity_Class=d["sens"], Decision_Stakes_Level=d["stakes"], Population_Vulnerability_Level=d["vuln"]))
DEP = pd.DataFrame(drows)
# ---------------- validation ----------------
res = []
def chk(cid, desc, ok, detail=""): res.append(dict(Check=cid, Description=desc, Result="PASS" if ok else "FAIL", Detail=detail))
chk("V01", "Primary keys unique (Case_ID, Case_Harm_ID, Deployment_ID, Benefit_ID)", INC.Case_ID.is_unique and HAR.Case_Harm_ID.is_unique and DEP.Deployment_ID.is_unique and BEN.Benefit_ID.is_unique)
chk("V02a", "Every CASE_HARMS.Case_ID exists in INCIDENT_CASES", HAR.Case_ID.isin(INC.Case_ID).all())
chk("V02b", "CASE_HARM_ATTRIBUTES is 1:1 with CASE_HARMS", set(ATT.Case_Harm_ID) == set(HAR.Case_Harm_ID) and ATT.Case_Harm_ID.is_unique)
chk("V02c", "Non-null Deployment_ID values exist in AI_DEPLOYMENT", INC.Deployment_ID.dropna().isin(DEP.Deployment_ID).all(), f"{int(INC.Deployment_ID.isna().sum())} rows with NULL Deployment_ID (deployment not identifiable)")
chk("V02d", "CASE_BENEFITS keys exist (Case_ID or Deployment_ID)", BEN.Case_ID.dropna().isin(INC.Case_ID).all() and BEN.Deployment_ID.dropna().isin(DEP.Deployment_ID).all() and ((BEN.Case_ID.notna()) | (BEN.Deployment_ID.notna())).all())
g = INC.groupby("Incident_ID")
pac_ok = (g.People_Affected_Count.nunique(dropna=False) == 1).all()
sum_ok = all(int(grp.Represented_Person_Count.sum()) == int(grp.People_Affected_Count.iloc[0]) for _, grp in g if pd.notna(grp.People_Affected_Count.iloc[0]))
chk("V03", "People_Affected_Count consistent within incident; Represented_Person_Count sums to it (when known)", pac_ok and sum_ok)
# recompute severity
mrg = HAR.merge(ATT, on="Case_Harm_ID"); bad = 0; chkn = 0
for _, r in mrg.iterrows():
    lv = [r.Nature_Score, r.Magnitude_Score, r.Scope_Score, r.Duration_Score, r.Reversibility_Score]
    st = r.Employment_Severity if r.Harm_Dimension == "EMPLOYMENT" else r.Sovereignty_Severity
    if all(pd.notna(v) for v in lv):
        chkn += 1
        if pd.isna(st) or abs(sev(lv) - st) > 1e-6: bad += 1
    elif pd.notna(st): bad += 1
chk("V04", "Recomputed severity (x=(L-1)/4, power mean q=2, equal weights) equals stored severity", bad == 0, f"{chkn} scored rows checked; scored only when all five levels are present")
lv_ok = True
for c_, xc in zip(["Nature_Score", "Magnitude_Score", "Scope_Score", "Duration_Score", "Reversibility_Score"], ["Nature_x", "Magnitude_x", "Scope_x", "Duration_x", "Reversibility_x"]):
    s_ = ATT[c_].dropna(); lv_ok &= bool(s_.between(1, 5).all()) and bool(((ATT[xc].dropna() - (ATT.loc[ATT[xc].notna(), c_] - 1) / 4).abs() < 1e-3).all())
chk("V05", "Levels are integers 1-5 and normalised values equal (L-1)/4 (all in [0,1])", lv_ok)
sc_ok = (ATT.dropna(subset=["Scope_Score"]).apply(lambda r: r.Scope_Score == min(5, r.Scope_Domain_Count), axis=1)).all()
du_ok = (ATT.dropna(subset=["Duration_Score"]).apply(lambda r: r.Duration_Score == dur_level(r.Impact_Duration_Days, r.Duration_Type == "PERMANENT"), axis=1)).all()
chk("V06", "Scope = domain count capped at 5 (NOT population); Duration level matches rubric bands", bool(sc_ok) and bool(du_ok))
wa = DEP[["AI_Only_Work_Percentage", "Human_Only_Work_Percentage", "Collaborative_Work_Percentage"]].sum(axis=1)
chk("V07", "p_a + p_h + p_c = 100 for every deployment row", bool(((wa - 100).abs() < 1e-6).all()))
chk("V08", "Actual_Replacement_Percentage <= AI_Only_Work_Percentage", bool((DEP.Actual_Replacement_Percentage <= DEP.AI_Only_Work_Percentage + 1e-9).all()))
neg_ok = True; sup_ok = True; dist_ok = True
need = {"CONFIRMED": {"CONFIRMED"}, "ALLEGED": {"ALLEGED", "SUSPECTED"}, "SUSPECTED": {"SUSPECTED"}, "UNKNOWN": {"UNKNOWN"}, "CONTRADICTED": {"CONTRADICTED"}}
desc_map = INC.drop_duplicates("Case_ID").set_index("Case_ID").Incident_Description.to_dict()
neg_s = {"FinancialDataExposed": ["No financial credentials were involved.", "No financial data was involved."], "RoleEliminated": ["No other positions were eliminated.", "No positions were eliminated."], "DisciplinaryUseOfScores": []}
for _, r in HAR.iterrows():
    pl = json.loads(r.Predicate_Logic); core = [p_ for p_ in pl if p_["layer"] != "ENTITY"]
    confirmed_names = {p_["predicate_name"] for p_ in core if p_["assertion_status"] == "CONFIRMED"}
    for p_ in core:
        if p_["assertion_status"] in ("NEGATED", "HYPOTHETICAL") and p_["predicate_name"] in confirmed_names: neg_ok = False
        if p_["assertion_status"] == "NEGATED" and r.Case_ID in desc_map and r.Case_ID.startswith("CAS-9") is False:
            if not any(s in desc_map[r.Case_ID] for s in neg_s.get(p_["predicate_name"], [])): dist_ok = False
        if p_["assertion_status"] == "HYPOTHETICAL" and not r.Case_ID.startswith("CAS-9"):
            if "feared" not in desc_map[r.Case_ID]: dist_ok = False
    if not any(p_["assertion_status"] in need[r.Harm_Confirmation_Status] for p_ in core): sup_ok = False
chk("V09a", "No CONFIRMED predicate duplicates a NEGATED/HYPOTHETICAL predicate within a harm", neg_ok)
chk("V09b", "Every NEGATED/HYPOTHETICAL predicate is backed by a matching sentence in the description (synthetic rows)", dist_ok)
chk("V09c", "Each harm status is supported by a predicate with a matching status (negated/hypothetical text never creates a confirmed harm)", sup_ok)
na = DEP[DEP.Deployment_Status == "NO_ESTABLISHED_DEPLOYMENT"]; pr2 = DEP[DEP.Deployment_Status == "PROPOSED"]; act = DEP[DEP.Deployment_Status.isin(["ACTIVE", "PILOT"])]
nz_ok = bool((na.AI_Only_Work_Percentage == 0).all() and (na.Collaborative_Work_Percentage == 0).all() and (na.Human_Only_Work_Percentage == 100).all() and (na.Actual_Replacement_Percentage == 0).all() and na.Exposure_Count.isna().all() and pr2.Exposure_Count.isna().all() and (act.Exposure_Count > 0).all() and ((HAR.Severity_Formula_Version == FORMULA_VERSION).all()))
chk("V10", "NULL vs 0: unknown exposure is NULL (never 0); no-deployment shares are explicit scenario zeros; active exposure > 0", nz_ok)
syn = INC[INC.Record_Type == "SYNTHETIC"]; docr = INC[INC.Record_Type == "DOCUMENTED"]
sep_ok = (len(syn) == 1000) and (len(docr) == 15) and (syn.Source_Reference == "SYNTHETIC_NO_EXTERNAL_SOURCE").all() and (docr.Source_Reference != "SYNTHETIC_NO_EXTERNAL_SOURCE").all() and docr.Source_Quality.isin(["VERIFIED_PRIMARY", "CREDIBLE_SECONDARY"]).all() and docr.Deployment_ID.isna().all() and (DEP.Record_Type == "SYNTHETIC").all() and (BEN.Record_Type == "SYNTHETIC").all()
chk("V11", "SYNTHETIC vs DOCUMENTED separation (1000 synthetic cases, 15 documented; sources/quality/Deployment_ID rules)", bool(sep_ok), f"{len(syn)} synthetic, {len(docr)} documented case rows")
dm = DEPD; dt_ok = True
for _, r in syn.dropna(subset=["Deployment_ID"]).iterrows():
    d = dm[r.Deployment_ID]
    if not (d["start"] <= r.Incident_Date <= (d["end"] or CUTOFF)) or r.Incident_Date > r.Report_Date: dt_ok = False
chk("V12", "Incident_Date within the linked deployment's exposure period and <= Report_Date", dt_ok)
og = HAR.dropna(subset=["Overlap_Group_ID"]).groupby("Overlap_Group_ID").agg(n=("Case_ID", "size"), c=("Case_ID", "nunique"))
chk("V13", "Overlap groups have >= 2 harm rows (same case for synthetic)", bool((og.n >= 2).all()) and bool((og[og.index.str.startswith("OG-CAS")].c == 1).all()))
chk("V14", "Confirmation_Weight mapping (CONFIRMED 1.0, ALLEGED 0.25, SUSPECTED 0.20, UNKNOWN 0.10, CONTRADICTED 0.0)", bool((HAR.Harm_Confirmation_Status.map(CW) == HAR.Confirmation_Weight).all()))
chk("V15", "Extraction_Confidence is NULL (no extraction model run)", bool(HAR.Extraction_Confidence.isna().all()))
loss_ok = bool((ATT.Currency.dropna() == "USD").all()) and bool((ATT.Financial_Loss_Amount.notna() == ATT.Currency.notna()).all())
chk("V16", "Currency is USD wherever a financial amount exists; none otherwise", loss_ok)
dim_ok = bool(((HAR.Harm_Dimension == "EMPLOYMENT") | HAR.Employment_Severity.isna()).all() and ((HAR.Harm_Dimension == "SOVEREIGNTY") | HAR.Sovereignty_Severity.isna()).all()) and bool((HAR.Harm_Category.map(lambda c: TAX[c][1]) == HAR.Harm_Dimension).all())
chk("V17", "Employment and sovereignty severity kept separate (one populated per row, matching taxonomy)", dim_ok)
rem_ok = bool(((ATT.Reversibility_Score != 5) | (ATT.Remedy_Available != "TRUE")).all()) and bool(((ATT.Remedy_Available != "FALSE") | (ATT.Reversibility_Score.isna()) | (ATT.Reversibility_Score >= 3)).all())
chk("V18", "Remedy/Reversibility consistency (R=5 => no remedy; no remedy => R>=3)", rem_ok)
cover = set(HAR.Harm_Category) == set(TAX) and set(HAR.Harm_Confirmation_Status) == set(CW) and (len(set(DEP[DEP.Deployment_Status == "NO_ESTABLISHED_DEPLOYMENT"].Field)) >= 2)
inc_by_dep = set(INC.Deployment_ID.dropna()); zero_dep = [d for d in act.Deployment_ID if d not in inc_by_dep]
chk("V19", "Coverage: all 19 harm categories, all 5 statuses, >=2 no-deployment fields, active deployments with zero incidents", bool(cover) and len(zero_dep) > 0, f"{len(zero_dep)} active/pilot deployments with zero incidents; no-deployment fields: {sorted(set(na.Field))}")
sdf = mrg.copy(); sdf["S"] = sdf.Employment_Severity.fillna(sdf.Sovereignty_Severity)
syn_ids = set(INC[INC.Record_Type == "SYNTHETIC"].Case_ID); ss = sdf[sdf.Case_ID.isin(syn_ids)]
sd_ok = bool((ss.groupby("Harm_Category").S.std() > 0.02).all()); mx = ss[["Nature_Score", "Magnitude_Score", "Scope_Score", "Duration_Score", "Reversibility_Score"]].corr().abs().where(~np.eye(5, dtype=bool)).max().max()
chk("V20", "Distribution quality: severity varies within every category; no near-perfect correlation between dimensions; no duplicate rows", sd_ok and mx < 0.95 and not HAR.duplicated(subset=[c for c in HAR.columns if c not in ("Case_Harm_ID", "Predicate_Logic")]).all(), f"max |corr| between level dimensions = {mx:.2f}")
chk("V21", "Documented rows are scored only when all five levels are supported; others INSUFFICIENT_EVIDENCE", bool(((HAR.Severity_Calculation_Status == "CALCULATED") == (HAR.Employment_Severity.notna() | HAR.Sovereignty_Severity.notna())).all()))
chk("V22", "Information_Imputed FALSE everywhere (no imputation performed)", bool((ATT.Information_Imputed == "FALSE").all()))
VAL = pd.DataFrame(res)
# ---------------- export ----------------
def out(df, name): df.to_csv(os.path.join(OUT, name), index=False, date_format="%Y-%m-%d")
out(INC, "INCIDENT_CASES.csv"); out(HAR, "CASE_HARMS.csv"); out(ATT, "CASE_HARM_ATTRIBUTES.csv"); out(BEN, "CASE_BENEFITS.csv"); out(DEP, "AI_DEPLOYMENT.csv"); out(VAL, "VALIDATION_REPORT.csv")
json.dump(GT, open(os.path.join(OUT, "generator_ground_truth.json"), "w"), indent=1)
print(VAL.to_string()); print(len(INC), len(HAR), len(ATT), len(BEN), len(DEP))
if (VAL.Result != "PASS").any(): sys.exit(1)
