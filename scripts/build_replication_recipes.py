"""Build replication_protocol/recipes.json: the selection rule, every candidate checked, and the selection.

Protocol v2 (2026-10-31). Candidate facts (licence, gate, revision SHA) were read from
https://huggingface.co/api/datasets/<id> and the dataset cards on 2026-10-31; this script records them and
applies the rule. It makes no network calls.
"""
import json
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SALT = "ASLA-replication-2026-10-31"
rule = {
  "fixed_on": "2026-10-31, before any training",
  "eligibility": {
    "E1_licence": "the Hugging Face dataset card carries an open licence tag (odc-by, cc0-1.0, cc-by-4.0, apache-2.0, mit) or its text states that all content is openly licensed or public domain, or defers to Common Crawl terms of use for crawled text; 'other' with no such statement fails",
    "E2_access": "the canonical repository is public and downloadable without manual approval (ungated or auto-gated); unofficial mirrors do not count",
    "E3_english_and_size": "an English portion is selectable from released metadata (a language field or a file/config split) and holds at least 30B tokens by the card's own figures, enough for the 300M run's 29.5B tokens without repetition",
    "E4_distinct_from_datadecide": "not one of DataDecide's corpora (C4, DCLM-Baseline, Dolma 1.6++/1.7, Falcon RefinedWeb, FineWeb-Edu, FineWeb-Pro) and at most 20% of its documents drawn from them or from the pools they were filtered from (DCLM-Pool, FineWeb, RefinedWeb, mC4/C4 family); judged from the card's stated composition; a card that does not state composition fails",
    "E5_general_domain": "a general-domain corpus (web or a multi-source mixture), not restricted to one domain (code, mathematics, scientific papers) or one source (a single encyclopedia or book collection) and not synthetic-only",
    "E6_one_per_family": "at most one corpus per release family (e.g. RedPajama-V2, RedPajama-1T and SlimPajama are one family; so are a corpus and its filtered or re-released variants); within a family, the member that meets E1 to E5 with the smallest salted crc32 key is kept"
  },
  "selection": "if more than 10 candidates are eligible, take the 10 with the smallest crc32('" + SALT + ":' + hf_id); the rest are ordered reserves. If 10 or fewer are eligible, all are selected.",
  "salt": SALT,
  "per_corpus_processing": "use the English portion as released, with no additional quality filtering; remove documents whose NFC text hash matches the evaluation set (replication_protocol/eval_set.jsonl, DataDecide's in-loop c4_en-validation file; scripts/decontaminate.py); shuffle once with the run's data-order seed; tokenize with the ladder tokenizer; never repeat a document within a run",
  "search": "candidates: (1) corpora known to the authors as public LM-pretraining corpora; (2) every dataset returned by the HF API listing filter=language:en, filter=task_categories:text-generation, sorted by downloads, top 100 for each size_categories value n>1T, 100B<n<1T, 10B<n<100B, 1B<n<10B, 100M<n<1B (322 datasets, 2026-10-31); listed datasets that are evidently mirrors, tokenized copies, subsets or filtered derivatives of FineWeb, FineWeb-Edu, DCLM, Dolma, RefinedWeb or C4, or that are not text-pretraining corpora (code, math, single-domain, synthetic, instruction, multimodal captions, personal uploads with no card), were screened out by name and card and are not listed individually; every other one is listed below with its outcome",
  "replacement_order": "if a selected corpus is unavailable or fails processing before training, it is replaced by the first unused reserve in the order given in 'reserves'; each replacement is recorded as a deviation and power is recomputed with the frozen simulator before training",
  "verified_with": "https://huggingface.co/api/datasets/<id> (cardData.license, gated, sha) and the dataset card README, fetched 2026-10-31",
}
C = [
 # id, hf_id, sha, licence, gated, english, size_evidence, composition_note, eligible(dict of failures), processing_note
 ("redpajama_v2","togethercomputer/RedPajama-Data-V2","aa033f6b6480c445557e9118283993403f069f6a","Common Crawl Foundation terms of use (card); code Apache-2.0","no","'en' config/partitions","card: 30T+ deduplicated tokens, English the largest language","own CCNet processing of 84 CC snapshots",[], "head_middle English partitions with the released exact-duplicate ids removed; quality signals not used"),
 ("redpajama_1t","togethercomputer/RedPajama-Data-1T","398f92572e94f4793e41c22ab7ea2a788d9e7de4","per subset: CC terms of use, C4 ODC-BY, GitHub MIT/BSD/Apache, others per card","no","English corpus","card: 1.2T tokens","CC (CCNet) about 878B, C4 about 175B (about 15%, under the 20% limit), GitHub, books, arXiv, Wikipedia, StackExchange",["E6_one_per_family"], None),
 ("txt360","IFM/TxT360","3337c47e3c09726092745da02aac0b6a7da055ed","odc-by","no","English","card: about 5T deduplicated tokens","own filtering of 99 CC snapshots plus 14 curated sources; card mentions FineWeb, Dolma, RedPajama only as comparisons; LLM360/TxT360 now redirects here",[], "released mixture (the card's weighting recipe is not applied)"),
 ("comma_v0_1","common-pile/comma_v0.1_training_dataset","5afc546db324e7f39f297ba757c9a60547151e7c","no tag; Common Pile is openly licensed text by construction (each source carries an open licence)","no","English","card: 463.6B tokens main stage","openly licensed sources; peS2o (a Dolma component) is 43.3B of 463.6B, about 9%",[], "main-stage sources at released proportions, without the card's repeat factors"),
 ("common_corpus","PleIAs/common_corpus","307910e4c5d040d6f318e6edf2a2b97849155771","no tag; card: all data uncopyrighted or freely licensed","no","per-document language field","card: 2.27T tokens, mostly English and French","books, newspapers, scientific, government and legal documents, code; no DataDecide source",[], "documents whose language field is English"),
 ("finepdfs","HuggingFaceFW/finepdfs","220bac3acbf07789502c621d2d33952f51ac7f86","odc-by","no","eng_Latn config","card: English is the largest config (over 1T tokens)","text extracted from PDFs in Common Crawl; disjoint from FineWeb's HTML extraction",[], "eng_Latn config"),
 ("matrix_cc_en","m-a-p/Matrix","c13c43bc3232816543751a61a46e9a23503361bd","apache-2.0","no","cc_en files (200 shards)","card: 4.69T tokens bilingual; cc_en is 200 of 496 shards","MAP-Neo's own CC processing",[], "cc_en shards only (the other components do not mark language)"),
 ("hplt_v2_en","HPLT/HPLT2.0_cleaned","d1324a5283f762ee62c2a5c81de08fc6450ea540","cc0-1.0 (packaging); underlying text under source terms","no","eng_Latn config","card: 8T tokens over 193 languages; the card's ablation samples 100B English tokens","Internet Archive and CC crawls, own pipeline",[], "eng_Latn config"),
 ("open_license_corpus","kernelmachine/open-license-corpus","384d5e19d19361803630ce4d382604267d3951d2","apache-2.0","no","primarily English","card: 228B tokens","public-domain and permissively licensed legal, code, science, conversation, books, news, encyclopedic text; uses RedPajama's GitHub/StackExchange/Wikipedia subsets, not DataDecide sources",[], "all domains at released proportions"),
 ("madlad_400_en","allenai/MADLAD-400","9d886a76bd8fa69b294f2dd3843dacb8388ee5a5","odc-by tag; card text CC-BY-4.0","no","en config","card: 1.84B English documents","Google's own document-level CC audit and filtering",[], "en clean split"),
 # excluded
 ("slimpajama","cerebras/SlimPajama-627B",None,None,"repository not accessible (API 401 on 2026-10-31)","English","627B","deduplicated RedPajama-1T",["E2_access","E6_one_per_family"],None),
 ("essential_web_v1","EssentialAI/essential-web-v1.0","ce4eccc7e9604667b6d7f32cb6274b8b41f3113d","odc-by","no","English","24T","the whole DCLM-Pool, annotated (card: 'DCLM Pool: 89 resiliparse-extracted CC snapshots')",["E4_distinct_from_datadecide"],None),
 ("zyda_2","Zyphra/Zyda-2","bc389cd62969a275b2e3c3e11db826ba1918411a","odc-by","no","English","5T","mostly DCLM and FineWeb-Edu",["E4_distinct_from_datadecide"],None),
 ("zyda_1","Zyphra/Zyda","d42bc68cfb537710a580e8d4b13ec536fcc151a6","odc-by","no","English","1.3T","RefinedWeb, C4, peS2o and others",["E4_distinct_from_datadecide"],None),
 ("smollm_corpus","HuggingFaceTB/smollm-corpus","3ba9d605774198c5868892d7a8deda78031a781f","odc-by","no","English","about 250B","mostly deduplicated FineWeb-Edu",["E4_distinct_from_datadecide"],None),
 ("fineweb","HuggingFaceFW/fineweb","9bb295ddab0e05d785b879661af7260fed5140fc","odc-by","no","English","15T","the pool FineWeb-Edu was filtered from",["E4_distinct_from_datadecide"],None),
 ("culturax","uonlp/CulturaX","6a8734bc69fefcbb7735f4f9250f43e4cd7a442e","no tag; card hidden behind auto gate","auto","en","large","mC4 plus OSCAR (C4 family)",["E4_distinct_from_datadecide"],None),
 ("nemotron_cc_v2","nvidia/Nemotron-CC-v2","2669787c66d18601c0e91167fd520be6c1245865","other (NVIDIA data agreement)","manual","English","multi-trillion","own CC processing",["E1_licence","E2_access"],None),
 ("oscar_2301","oscar-corpus/OSCAR-2301","c2930464071faebadfb5491e2b4f99666da7a1e6","cc0-1.0","manual","en","large","own CC processing",["E2_access"],None),
 ("pile_uncopyrighted","monology/pile-uncopyrighted","3be90335b66f24456a5d6659d9c8d208c0357119","other, no licence statement","no","English","under 300B","The Pile minus Books3, BookCorpus2, OpenSubtitles, YTSubtitles, OWT2",["E1_licence"],None),
 ("openwebtext","Skylion007/openwebtext","79d93d786212f7344586290adb811d4ae6a1762c","cc0-1.0","no","English","about 9B tokens","Reddit-linked web pages",["E3_english_and_size"],None),
 ("cosmopedia","HuggingFaceTB/cosmopedia","0ae6ec63f91742bd2d1eaef4f02232c55d719385","apache-2.0","no","English","about 25B tokens","synthetic textbooks",["E3_english_and_size","E5_general_domain"],None),
 ("wikipedia","wikimedia/wikipedia","b04c8d1ceb2f5cd4588862100d08de323dccfbaa","cc-by-sa-3.0, gfdl","no","20231101.en","under 10B tokens","a single encyclopedia",["E3_english_and_size","E5_general_domain"],None),
 ("synth","PleIAs/SYNTH","0d6813a2966662c39f22f0b9af28a0c1c9f7a437","cc-by-4.0","no","English","","synthetic, from Wikipedia",["E5_general_domain"],None),
 ("institutional_books","institutional/institutional-books-1.0","1f12e87e317077474679899a8f78feaeb8a995ff","no tag","auto","multilingual","242B","a single library's books",["E1_licence","E5_general_domain"],None),
 ("industry_corpus_2","BAAI/IndustryCorpus2","249ea52f1ee412303e9741ae69fdd2109c0f8e1c","apache-2.0","auto","en and zh","","composition sources not stated in the public card",["E4_distinct_from_datadecide"],None),
 ("tiger_pretrain_en","TigerResearch/pretrain_en","916f8042e26f138ba73a12ec0a13cc043a2ef720","apache-2.0","no","English","token count not stated","composition not stated",["E3_english_and_size","E4_distinct_from_datadecide"],None),
 ("mint_1t_html","mlfoundations/MINT-1T-HTML","906a8b85cea61198ff7339c4dd711ad0b5361847","cc-by-4.0","no","English (fastText language ID)","card: 742B text tokens (data_v1_1)","HTML from CC WARC dumps 2017 to 2024, own pipeline",[], "data_v1_1 text segments concatenated in document order, images dropped"),
 ("cc100_en","statmt/cc100","8c658c983d32eab9170d77d416252cfaa0c23e96","no tag ('unknown'); card: no IP claim, Common Crawl terms of use apply","no","en config","card: English 82G compressed; the XLM-R paper (Conneau et al. 2020) reports 55.6B English tokens","CCNet processing of 2018 CC snapshots (not the 2019 snapshot C4 was built from)",[], "en config"),
 ("obelics","HuggingFaceM4/OBELICS","08b7dca61ec09bacd68234ba6c388209b126d31c","cc-by-4.0","no","English","card: 141M English documents, 115B text tokens","interleaved web documents from CC dumps, own pipeline",[], "text segments concatenated in document order, images dropped"),
 ("omnicorpus_cc","OpenGVLab/OmniCorpus-CC","8038991c62db064e5469a8a509261d8f4e85fd61","cc-by-4.0","no","English (card language en)","card: 988M documents, about 1.7T text tokens over the corpus","interleaved documents from CC dumps 2013 to 2023, own pipeline",[], "text segments concatenated in document order, images dropped"),
 ("mint_1t_pdf_2023_06","mlfoundations/MINT-1T-PDF-CC-2023-06","","cc-by-4.0","no","English","card gives no token count for this subset","PDFs from CC WAT, MINT-1T family",["E3_english_and_size"],None),
 ("omnicorpus_cc_210m","OpenGVLab/OmniCorpus-CC-210M","39c856327b0c18a89915fa4c94ae870d94e18601","cc-by-4.0","auto","en","card hidden behind the gate","OmniCorpus family",["E3_english_and_size","E4_distinct_from_datadecide"],None),
 ("cc100_documents_mirror","singletongue/cc100-documents","","no tag","no","multilingual","","unofficial document-level re-release of CC-100",["E2_access"],None),
 ("txt360_3efforts","LLM360/TxT360-3efforts","bfc4a082d11967cd7810fe0b773be87bf54fb32e","cc-by-4.0","no","English","","TxT360 family; larger crc32 key than IFM/TxT360",["E6_one_per_family"],None),
 ("finepdfs_edu","HuggingFaceFW/finepdfs-edu","9cfabe2127faca99b3d5c4dc6d1fcb397399ebde","odc-by","no","eng_Latn","","FinePDFs family; larger crc32 key",["E6_one_per_family"],None),
 ("hplt_v1_2","HPLT/hplt_monolingual_v1_2","","cc0-1.0","no","en","","HPLT family; larger crc32 key",["E6_one_per_family"],None),
 ("culturay","Viet-Mistral/CulturaY","","cc-by-4.0","no","multilingual","","built from HPLT; HPLT family",["E6_one_per_family"],None),
 ("common_pile_raw","common-pile/raw_v0.1_parquet","","no tag","no","English","","Common Pile family; larger crc32 key than Comma",["E6_one_per_family"],None),
 ("cccc_all_domains","nkandpa2/cccc_all_domains","930b8e9b46b55730b12675e643e9778fa87f58af","odc-by","auto","English","","Common Pile's CC Creative Commons component; Common Pile family",["E6_one_per_family"],None),
 ("colossal_oscar_1","oscar-corpus/colossal-oscar-1.0","6e7659d7b54f899942841fd046251ce8d9564882","cc0-1.0","auto","multilingual","card hidden behind the gate","OSCAR family",["E3_english_and_size","E4_distinct_from_datadecide"],None),
 ("oscar_2019","oscar-corpus/oscar","65dd1dc9b23d5f2aa283e055e5edff5abbed995a","cc0-1.0","manual","en","","OSCAR family",["E2_access"],None),
 ("c5_creative_commons","BramVanroy/CommonCrawl-CreativeCommons","fa6453d0f267675babb06b1968928b81b47e712c","cc (mixed Creative Commons licences, including NC)","no","eng config","card states no token count","CC pages linking a Creative Commons licence",["E1_licence","E3_english_and_size"],None),
 ("cc_news","stanford-oval/ccnews","d733e654c9a506df519e1a166a86c118c7657ce4","no tag","no","multilingual","","news articles only",["E5_general_domain"],None),
 ("croissant","croissantllm/croissant_dataset","4309f6cd724528be5ae590dd3552f82d834ad00e","no tag; per-source licences in the technical report","no","en and fr","","English composition not stated in the card",["E4_distinct_from_datadecide"],None),
 ("luciole","OpenLLM-France/Luciole-Training-Dataset","dc89cbe9b500eb934f198e7b4718926132238117","cc-by-sa-4.0","no","multilingual","","includes Dolma 3 and FineWeb2 subsets",["E4_distinct_from_datadecide"],None),
 ("eaglex","RWKV/EagleX-WorldContinued","b3c23a3c57f5902c0abb65470a34d68820e45416","cc-by-sa-4.0","no","multilingual","","a mixture citing FineWeb among its sources",["E4_distinct_from_datadecide"],None),
 ("crisp_165b","juliannunezb/crisp-165b","c90ea4ecdda5421c6452303d2e76aadda10dcf1b","odc-by","no","English","165B","DCLM-baseline is a co-equal web source",["E4_distinct_from_datadecide"],None),
 ("domyn_edge_mix","domyn/Domyn-Edge-Pretraining-Mix-8.4T","b0eb2e99ae817ecd40959c92692f71a0e0911162","apache-2.0","no","multilingual","","web crawl portion is FineWeb",["E4_distinct_from_datadecide"],None),
 ("tnsa_pt_hf500b","TNSA/PT-HF500B","cd72cef5badec2ffdd8d33e52d4c08c0e65321fe","odc-by","no","English","about 486B generated tokens","synthetic",["E5_general_domain"],None),
 ("opentransformer_crawl","OpenTransformer/web-crawl-2026","f24960eda224cb2a2d176b1f4279639bf0bd4801","apache-2.0","no","not stated","card states no token count","own crawler seeded from the CC URL index",["E3_english_and_size"],None),
 ("plm_pretrain","PLM-Team/Pretrain-Dataset","3a7d4c175e8bfcf0cd17f78860a8d6056a7b02ab","mit","no","not stated","","composition not stated",["E4_distinct_from_datadecide"],None),
 ("climblab","OptimalScale/ClimbLab","","apache-2.0 (mirror); NVIDIA original cc-by-nc-4.0","no","English","","re-release of NVIDIA ClimbLab, built from Nemotron-CC and SmolLM-Corpus (FineWeb-Edu)",["E1_licence","E2_access","E4_distinct_from_datadecide"],None),
 ("bcai_cci4","BAAI/CCI4.0-M2-Base-v1","f8d5c95fb101c0d1d4e49a43d75182a71bdca3ff","no tag","auto","zh and en","","English composition not stated in the public card",["E4_distinct_from_datadecide"],None),
 ("the_pile","EleutherAI/pile","","other","no","English","","includes Books3; no open licence statement",["E1_licence"],None),
]
cands=[]
for c in C:
    d=dict(id=c[0],hf_id=c[1],revision_sha=c[2],licence=c[3],gated=c[4],english_portion=c[5],size_evidence=c[6],composition=c[7],fails=c[8],eligible=not c[8])
    if c[9]: d["processing"]=c[9]
    d["crc32_rank_key"]=zlib.crc32(f"{SALT}:{c[1]}".encode())
    cands.append(d)
elig=sorted([c for c in cands if c["eligible"]], key=lambda c:c["crc32_rank_key"])
sel, res = elig[:10], elig[10:]
out={"rule":rule,"n_considered":len(cands),"n_eligible":len(elig),"selected":[{k:c[k] for k in ("id","hf_id","revision_sha","licence","processing")} for c in sel],"reserves":[{k:c[k] for k in ("id","hf_id","revision_sha","licence","processing")} for c in res],
 "protocol_version":"v2 (2026-10-31): RedPajama-1T dropped under E6 (RedPajama family, RedPajama-V2 kept by crc32); search extended to the HF listing; TxT360 recorded under its canonical id IFM/TxT360",
 "resemblance_caveat":"among the selected, comma_v0_1 and open_license_corpus are openly licensed multi-domain mixtures that differ from DataDecide's web-heavy recipes more than the others do, and mint_1t_html and omnicorpus_cc are interleaved multimodal web corpora used for their text only; the predicted magnitude assumes resemblance and is weakest for these",
 "candidates":cands}
(REPO / "replication_protocol" / "recipes.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(len(cands), len(elig), [c["id"] for c in sel], [c["id"] for c in res])
