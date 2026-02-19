:- module(olimpics_rules, [ 
    olympic_advice/4,
    explain_verdict/4
]).

/** <module> Knowledge Base per Predizione Medaglie Olimpiche

    Questo modulo implementa un sistema esperto ibrido. Integra le predizioni
    probabilistiche di un modello Random Forest con una base di conoscenza
    simbolica per fornire analisi contestualizzate e spiegabili.
*/

% --- 1. FATTI ---

/** historical_elite(?NOC, ?Sport, -Dominance)
    Rappresenta le nazioni leader e il loro peso nel medagliere storico.
*/
historical_elite('ITA', 'Fencing', 20.12).
historical_elite('USA', 'Swimming', 34.80).
historical_elite('USA', 'Golf', 70.69).
historical_elite('CHN', 'Table Tennis', 45.61).
historical_elite('JAM', 'Athletics', 30.00).
historical_elite('JPN', 'Judo', 38.00).
historical_elite('BRA', 'Football', 25.00).

% --- 2. REGOLE ---

/** is_superpower(+NOC)
    Vero se la nazione possiede una struttura d'élite in più settori.
*/
is_superpower(NOC) :- 
    findall(S, historical_elite(NOC, S, _), L),
    length(L, N), N >= 2.

/** sector_type(+Sport, -Type)
    Definisce la fluidità di un settore in base alla dominanza del leader.
*/
sector_type(Sport, closed) :- 
    historical_elite(_, Sport, Dominance), Dominance > 40.0, !.
sector_type(_, open).



/** olympic_advice(+Probability, +NOC, +Sport, -Advice)
    Predicato principale che genera il verdetto finale.
*/

% Caso A: Eccellenza Storica Confermata (ML e KB concordano)
olympic_advice(Prob, NOC, Sport, Advice) :-
    Prob >= 0.70,
    historical_elite(NOC, Sport, Dom),
    atomic_list_concat(['SUCCESSO ATTESO: Dominio storico confermato (Dominanza leader: ', Dom, '%).'], Advice), !.

% Caso B: Tradizione Resiliente (La storia compensa dati ML incerti)
olympic_advice(Prob, NOC, Sport, "TRADIZIONE COMPETITIVA: La solida storia nazionale sostiene l'atleta nonostante i dati statistici incerti.") :-
    Prob >= 0.45,
    historical_elite(NOC, Sport, _), !.

% Caso C: Nuova Potenza (Exploit in ascesa in settori fluidi)
olympic_advice(Prob, NOC, Sport, "NUOVA POTENZA: Trend statistico eccellente in un settore competitivo e fluido.") :-
    Prob >= 0.70,
    \+ historical_elite(NOC, Sport, _),
    sector_type(Sport, open), !.

% Caso D: Exploit Difficile (Dato ML alto contrastato da egemonia storica altrui)
olympic_advice(Prob, NOC, Sport, "EXPLOIT DIFFICILE: Il modello è ottimista, ma il settore è storicamente blindato da leader dominanti.") :-
    Prob >= 0.70,
    \+ historical_elite(NOC, Sport, _),
    sector_type(Sport, closed), !.

% Caso E: Scommessa (Atleta promettente senza radici sistemiche)
olympic_advice(Prob, _, _, "SCOMMESSA: Segnali promettenti, ma contesto privo di radici storiche consolidate.") :-
    Prob >= 0.45, !.

% Caso F: Fallback (Bassa probabilità e assenza di supporto storico)
olympic_advice(_, _, _, "SFIDA ESTREMA: Scarsa evidenza statistica e storica per una posizione di podio.").

% --- GENERAZIONE MOTIVAZIONI ---

/** explain_verdict(+Probability, +NOC, +Sport, -Reasons)
    Genera una lista di stringhe che motivano logicamente la decisione.
*/
explain_verdict(Prob, NOC, Sport, Reasons) :-
    findall(M, (
        (Prob > 0.70 -> M = 'Solidità statistica del modello ML');
        (historical_elite(NOC, Sport, _) -> M = 'Forte tradizione storica nazionale');
        (is_superpower(NOC) -> M = 'Appartenenza a nazione leader (Superpower)');
        (sector_type(Sport, open), Prob >= 0.45 -> M = 'Settore competitivo aperto a nuovi talenti');
        (Prob < 0.45 -> M = 'Performance storiche e recenti insufficienti')
    ), Reasons),
    (Reasons = [] -> Reasons = ['Analisi basata su parametri standard'] ; true).