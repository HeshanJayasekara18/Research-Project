"""PP1 research demonstration UI: truthful baselines and real YouTube comments."""
import csv
import json
from pathlib import Path
from functools import lru_cache
import gradio as gr
from retrieval import (ROOT, DATA, RetrievalEngine, NORMALIZED_METHODS, NORM_BM25_METHOD, NORM_LABSE_METHOD,
                       MULTIVIEW_METHODS, MV_BM25_METHOD, MV_LABSE_METHOD, MV_HYBRID_METHOD, SEMANTIC_METHOD)
from evaluate_crossscript import script_of
from retrieval import tokenize

ALL_METHODS=['BM25 (lexical)','Character TF-IDF',SEMANTIC_METHOD,NORM_BM25_METHOD,NORM_LABSE_METHOD,
             MV_BM25_METHOD,MV_LABSE_METHOD,MV_HYBRID_METHOD]
COMPARE_DEFAULT=['BM25 (lexical)',SEMANTIC_METHOD,NORM_BM25_METHOD,MV_BM25_METHOD,MV_HYBRID_METHOD]


@lru_cache(maxsize=1)
def engine():
    return RetrievalEngine()


def profile():
    return json.loads((ROOT/'results'/'dataset_profile.json').read_text(encoding='utf-8'))


def pairs_preview():
    rows=[]
    with (DATA/'seed_pairs_linked.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            rows.append([r['singlish_query'],r['seed_document_text'][:200],r['source_match_status']])
            if len(rows) == 75: break
    return rows


def query_examples():
    queries=[]
    with (DATA/'seed_pairs_linked.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['source_match_status']=='exact_text' and r['singlish_query'] not in queries:
                queries.append(r['singlish_query'])
            if len(queries)>=12: break
    return queries


GLOBAL_SCOPE='All videos (global research corpus)'


def scope_choices():
    e=engine()
    return [(GLOBAL_SCOPE,'')]+[(f"{v} · {len(e.video_rows[v])} comments",v) for v in e.video_ids()]


def scope_info(video_id):
    """Markdown describing what will be searched."""
    e=engine()
    if not video_id:
        return (f"**Scope:** global research corpus · **{len(e.docs):,}** comments from {len(e.video_ids())} videos searched. "
                'Use this mode for research experiments.')
    rows=e.video_rows.get(video_id,[])
    scripts={'sinhala':0,'latin':0}
    for i in rows:
        scripts[script_of(e.texts[i])]+=1
    url=e.docs[rows[0]]['youtube_video_url'] if rows else ''
    return (f"**Scope:** video `{video_id}` · **{len(rows)}** comments searched "
            f"(mostly Sinhala script: {scripts['sinhala']}, mostly Latin script/Singlish/English: {scripts['latin']}; no comment is excluded by script) · "
            f"[Open video]({url})\n\n"
            'Video-scoped retrieval only: comments from other videos are never scored. '
            'This prototype has **no creator login or channel-ownership verification**.')


CANDIDATE_NOTE=('**These are retrieved candidate comments, not human-verified evidence.** Ranking reflects similarity to the '
                'question, not confirmed relevance. Comments that share only generic words with the question (e.g. *video*, '
                '*viewers*) are not evidence for a specific question. This prototype does **not** generate answers or verify claims.')


def no_evidence_note(video_id,searched,method=''):
    where=f"from video `{video_id}` ({searched} comments searched)" if video_id else f"from the global corpus ({searched:,} comments searched)"
    return (f"**No matching comments were retrieved {where} using {method or 'this method'}.** "
            'This does **not** mean that no relevant comments exist: lexical methods need shared words or characters, '
            'and normalisation can fail. Try another method (e.g. LaBSE or multi-view) or rephrase the question.')


def shared_words(query,text):
    """Words the comment shares with the original or normalised query (display aid only)."""
    q=set(tokenize(query))|set(tokenize(engine().normalize(query).normalized))
    w=sorted(q&set(tokenize(text)))
    return ', '.join(w) if w else '(none: non-lexical match)'


def run_search(query,method,k,video_id=None):
    video_id=video_id or None
    try:
        searched=len(engine().video_rows.get(video_id,[])) if video_id else len(engine().docs)
        if method in MULTIVIEW_METHODS and str(query).strip():
            table,note=run_multiview(str(query),method,int(k),video_id)
            if not table:
                return [],no_evidence_note(video_id,searched,method)+'\n\n'+note
            return table,scope_info(video_id)+'\n\n'+CANDIDATE_NOTE+'\n\n'+note
        results = engine().search(str(query),method,int(k),video_id)
        table=[]
        for i,(d,score) in enumerate(results,1):
            table.append([i,round(score,4),d['comment_text'][:600],d['video_id'] or 'unknown',
                          d['youtube_video_url'],d['document_id'],shared_words(str(query),d['comment_text'])])
        if not results:
            return [],no_evidence_note(video_id,searched,method)
        head=scope_info(video_id)+'\n\n'+CANDIDATE_NOTE+'\n\n'
        if method in NORMALIZED_METHODS:
            return table,head+normalization_note(query)
        if method in (SEMANTIC_METHOD,):
            return table,head+('Dense retrieval results from pretrained LaBSE. Dense methods always return the nearest comments, '
                               'even when none is relevant. Relevance is not guaranteed and requires human evaluation.')
        return table,head+'Baseline lexical results only. Scores are method-specific and are NOT relevance judgments. Links open the VIDEO, not the individual comment.'
    except Exception as exc:
        return [],f'{type(exc).__name__}: {exc}'


def run_multiview(query,method,k,video_id=None):
    hits,info=engine().multiview(query,method,k,video_id=video_id)
    table=[[i,round(h.rrf_score,5),d['comment_text'][:600],d['video_id'] or 'unknown',d['youtube_video_url'],d['document_id'],
            shared_words(query,d['comment_text'])] for i,(d,h) in enumerate(hits,1)]
    views=info['views']
    lines=[f"**Multi-view RRF (experimental)** · k={info['k']} · top-{info['depth']} candidates per view · views: {', '.join(views)}",'']
    if not info['normalized_view_used']:
        lines+=[f"⚠️ **Normalisation not used** (status: {info['normalization'].status}). "
                "Fell back to original-query retrieval only; no second representation was invented.",'']
    else:
        lines+=[f"**Normalised view query:** {info['normalization'].normalized} · {info['normalization'].summary()}",'']
    if not hits:
        return [],'\n'.join(lines)+'\nNo view returned any candidate for this query.'
    lines+=['| Rank | RRF score | '+' | '.join(f'{v} rank (raw score)' for v in views)+' |','|---:|---:|'+'---:|'*len(views)]
    for i,(d,h) in enumerate(hits,1):
        cells=[f"{h.views[v][0]} ({h.views[v][1]:.3f})" if v in h.views else '—' for v in views]
        lines.append(f"| {i} | {h.rrf_score:.5f} | "+' | '.join(cells)+' |')
    lines+=['','RRF uses only ranks, so BM25 and cosine scores are never compared directly. Raw scores are per-view and '
            "NOT relevance judgments. '—' = not among that view's top candidates."]
    return table,'\n'.join(lines)


def compare_methods(query,k,methods,video_id=None):
    video_id=video_id or None
    query=str(query).strip()
    if not query:
        return 'Enter a query.','',[]
    if not methods:
        return 'Select at least one method.','',[]
    k=int(k); ranks={}; errors=[]
    for m in methods:
        try:
            ranks[m]={d['document_id']:i for i,(d,_) in enumerate(engine().search(query,m,k,video_id),1)}
        except Exception as exc:
            errors.append(f"- {m}: {type(exc).__name__}: {exc}")
    norm=engine().normalize(query)
    head=[scope_info(video_id),'',f"**Normalisation:** {norm.summary()}  \n**Normalised query:** {norm.normalized or '—'}"]
    if errors:
        head+=['','**Methods that failed:**']+errors
    names=list(ranks)
    union=set().union(*(set(r) for r in ranks.values())) if ranks else set()
    ov=['| Method | Returned | Unique to this method | '+' | '.join(f'shared with #{j+1}' for j in range(len(names)))+' |',
        '|---|---:|---:|'+'---:|'*len(names)]
    for m in names:
        if not ranks[m]:
            head+=['',no_evidence_note(video_id,len(engine().video_rows.get(video_id,[])) if video_id else len(engine().docs),m)]
    for i,m in enumerate(names):
        others=set().union(*(set(ranks[o]) for o in names if o!=m))
        ov.append(f"| #{i+1} {m} | {len(ranks[m])} | {len(set(ranks[m])-others)} | "+
                  ' | '.join('—' if o==m else str(len(set(ranks[m])&set(ranks[o]))) for o in names)+' |')
    ov+=['',f"**{len(union)}** distinct comments across {len(names)} methods (top-{k} each). "
         'Ranks are shown instead of raw scores because scores from different methods are not on a common scale.']
    rows=[]
    for doc_id in union:
        d=engine().by_id[doc_id]; r=[ranks[m].get(doc_id) for m in names]
        rows.append([sum(x is not None for x in r)]+[x if x is not None else '—' for x in r]+
                    [script_of(d['comment_text']),d['comment_text'][:300],shared_words(query,d['comment_text']),d['video_id'],doc_id])
    rows.sort(key=lambda row:(-row[0],min(x for x in row[1:1+len(names)] if x!='—'),row[-1]))
    headers=['# methods']+[f'#{i+1} rank' for i in range(len(names))]+['Script','Comment (300 chars)','Shared words with query','Video ID','Document ID']
    return '\n'.join(head),'\n'.join(ov),gr.update(value=rows,headers=headers)


def normalization_note(query):
    r=engine().normalize(str(query))
    rows=' · '.join(f"`{t.source}`→{t.output} *({t.status})*" for t in r.tokens)
    return (f"**Normalised query (experimental, {r.method}):** {r.normalized}\n\n"
            f"**Status:** {r.summary()}\n\n{rows}\n\n"
            "Unresolved words are kept in Latin script, never guessed. 'ambiguous' = close alternatives existed. "
            "The lexicon is this corpus's Sinhala vocabulary; this is NOT a general transliteration model. "
            "Scores are method-specific and are NOT relevance judgments.")


def show_diagnostic():
    path=ROOT/'results'/'seed_diagnostic.json'
    if not path.exists():
        return 'Run `python evaluate_seed.py` to generate exploratory results.'
    d=json.loads(path.read_text(encoding='utf-8'))
    lines=[f"**Eligible unique queries:** {d.get('eligible_unique_queries',0)}",
           '**This is a seed-pair diagnostic, NOT independent nDCG or Recall.**',
           'Queries were manually reviewed after AI suggestion and may have been written from the documents. '
           'Only exact-text-linked positive pairs are considered. Unjudged comments cannot be assumed irrelevant.', '',
           '| Method | Known-positive Hit@1 | Known-positive Hit@10 |',
           '|---|---:|---:|']
    for name,v in d.get('methods',{}).items():
        lines.append(f"| {name} | {v['known_positive_hit_at_1']:.1%} | {v['known_positive_hit_at_10']:.1%} |")
    return '\n'.join(lines)


prof=profile()
header=f"""# Singlish Retrieval Engine
### Progress Presentation 1 · Working Research Prototype
**{prof['clean_corpus_rows']:,}** unique video/text records · **{prof['distinct_valid_videos']}** video IDs · **{prof['seed_pairs']}** seed query–comment pairs

Explore **real Sinhala and Singlish YouTube comments**, compare simple retrieval baselines, and inspect the comment dataset. Results are retrieved candidates, not verified evidence.
"""

with gr.Blocks(title='Singlish Retrieval Engine · PP1') as demo:
    gr.Markdown(header)
    with gr.Tab('🔎 Live Retrieval'):
        with gr.Row():
            scope=gr.Dropdown(scope_choices(),value='',label='Search scope: one YouTube video, or the global research corpus',scale=3)
        scope_md=gr.Markdown(scope_info(None))
        scope.change(scope_info,inputs=[scope],outputs=[scope_md])
        with gr.Row():
            q=gr.Textbox(label='Ask in Singlish',placeholder='e.g., slt internet eka gana minissu mokakda kiyanne?',lines=2,scale=4)
            method=gr.Dropdown(ALL_METHODS,
                               label='Retrieval method',value='BM25 (lexical)',scale=2)
        with gr.Row():
            count=gr.Slider(3,20,step=1,value=8,label='Top results')
            search_btn=gr.Button('Retrieve candidate comments',variant='primary')
        gr.Examples(examples=query_examples()[:8],inputs=[q],label='Try queries from the handmade seed dataset')
        note=gr.Markdown('Choose a method and enter a Singlish query. Some cross-script queries will fail using lexical retrieval, which is precisely the research problem.')
        results=gr.Dataframe(headers=['Rank','Score','Original comment (up to 600 chars)','Video ID','Video URL','Prototype document ID','Shared words with query'],datatype=['number','number','str','str','str','str','str'],interactive=False,wrap=True,label='Retrieved candidate comments (NOT human-verified evidence)')
        search_btn.click(run_search,inputs=[q,method,count,scope],outputs=[results,note])
        q.submit(run_search,inputs=[q,method,count,scope],outputs=[results,note])
        gr.Markdown('**Source limitation:** Your existing CSV does not contain comment IDs. The video links are verifiable video-level references, not precise comment permalinks.')
    with gr.Tab('⚖️ Method Comparison'):
        gr.Markdown('Run the **same query** through several methods and compare **ranks** side by side. '
                    'Overlap shows which comments the methods agree on. Experimental methods are labelled; nothing here is a relevance judgment.')
        cscope=gr.Dropdown(scope_choices(),value='',label='Search scope: one YouTube video, or the global research corpus')
        cscope_md=gr.Markdown(scope_info(None))
        cscope.change(scope_info,inputs=[cscope],outputs=[cscope_md])
        with gr.Row():
            cq=gr.Textbox(label='Singlish query',lines=2,scale=4)
            ck=gr.Slider(5,20,step=1,value=10,label='Top-K per method',scale=1)
        cm=gr.CheckboxGroup(ALL_METHODS,value=COMPARE_DEFAULT,label='Methods to compare')
        cbtn=gr.Button('Compare methods',variant='primary')
        gr.Examples(examples=['Kana bath tikata kelawapu eka gana banala comments thiyenawada?',
                              'Kaha bath ekata kurudu pothu (cinnamon) danna kiyala minissu upades denawada?',
                              'Anuradhapura paththe oyage products ganna thiyenawada?'],inputs=[cq],label='Demo queries')
        cnorm=gr.Markdown(); cover=gr.Markdown()
        ctable=gr.Dataframe(interactive=False,wrap=True,label='Union of retrieved candidate comments (rank per method; — = not retrieved; NOT verified evidence)')
        cbtn.click(compare_methods,inputs=[cq,ck,cm,cscope],outputs=[cnorm,cover,ctable])
        cq.submit(compare_methods,inputs=[cq,ck,cm,cscope],outputs=[cnorm,cover,ctable])
    with gr.Tab('📚 Corpus + Handmade Queries'):
        gr.Markdown(f"""**Raw rows:** {prof['original_rows']:,} · **Cleaned records:** {prof['clean_corpus_rows']:,}

**Matched seed pairs (exact comment text):** {prof['seed_pairs_exact_text_matched']}/{prof['seed_pairs']} · **Unmatched seed pairs:** {prof['seed_pairs_unmatched']}

**Data limitations:** {prof['same_video_text_duplicates']:,} repeated video/text records collapsed; {prof['empty_comments']} empty comments excluded. The original file lacks real comment IDs, parent–reply IDs, and verified relevance judgments. Some comments may share identical text yet be distinct real comments; the prototype deduplicates text for indexing, not for archival truth.

**Status:** topic/domain labels still require validation. AI-assisted suggestions with human review are not independent relevance labels.
""")
        gr.Dataframe(value=pairs_preview(),headers=['Manually reviewed Singlish query','Paired YouTube comment','Exact-source match'],
                     datatype=['str','str','str'],interactive=False,wrap=True,label='First 75 query–comment examples')
    with gr.Tab('📊 Preliminary Baseline Diagnostic'):
        gr.Markdown(show_diagnostic())
        _mv=ROOT/'results'/'multiview_eval.md'
        gr.Markdown(_mv.read_text(encoding='utf-8').replace('# Milestone 3','### Milestone 3',1) if _mv.exists()
                    else 'Run `python evaluate_multiview.py` to generate the multi-view comparison.')
        gr.Markdown('**Research hypothesis:** Cross-script queries can have low term overlap with Sinhala evidence. Exploratory seed diagnostics suggest normalisation helps Sinhala-script targets and multi-view fusion balances both scripts; this must be confirmed on a separately human-annotated benchmark before any claim is made.')
        gr.Markdown('**PP1 evaluation milestone:** independent relevance judgments, verified sources, query-family-disjoint data splits, and nDCG@10 / Recall@50 on the same held-out test set.')
    with gr.Tab('🧭 Planned Research Pipeline'):
        gr.Markdown('''1. **Implemented baselines:** YouTube comment corpus + manual/AI-assisted seed queries + BM25, character TF-IDF, zero-shot LaBSE.
2. **Implemented (experimental):** corpus-lexicon-guided Singlish → Sinhala normalisation (lexicon-rule-v1) and multi-view Reciprocal Rank Fusion of original + normalised query views. Evaluated only with exploratory seed diagnostics.
3. **Next:** independent graded relevance annotation (nDCG@10, Recall@50); collect comment IDs, replies, timestamps; contrastive fine-tuning of the dense encoder; ablations.
4. **Later:** thread-aware evidence construction → Sinhala RAG answer → claim-level NLI verification + citations.

**Keep claims separate:** the prototype is a baseline research artifact; RAG and claim verification are not implemented in this version.''')

if __name__=='__main__':
    demo.launch(server_name='127.0.0.1',inbrowser=True,theme=gr.themes.Soft(primary_hue='blue'))
