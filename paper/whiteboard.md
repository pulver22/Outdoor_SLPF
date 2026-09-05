(just as a reminder. delete after usage or creation of an example document?)

## Do's and Dont's

- No references as nouns: 
   - **Bad:** \cite{wang2018cvpr} extracts point-
wise features using a PointNet, ...
   - **Good:** (The approach of) Wang et al.~\cite{wang2018cvpr} extracts point-
wise features using a PointNet, ...
- Capitalization
  - don't capitalize unless it is the name of a person
  - But capitalize headings (but not thing words)
  - legend: always lower-case
- No italics unless you want to interrupt the reading flow (e.g. to emphasize -> use not more then two times in a paragraph)
- Non-breaking space (`~`)
  - In front of formulas: `the width is computed as~$w=h^2$`
  - before cites: `Stachniss \etal~\cite{stachniss2004icra}`
  - before introducing abbreviations: `simultaneous localization and mapping~(SLAM)`
- use macros for \etal, \eg, \ie
- Figures should always top: `\begin{figrue}[t]`
- Units: 22\,cm
- no math ($) unless it is an equation.
- text in math mode should use `\text`, e.g., $A_{\text{tree}}$
- Mathematical notation: vector bold, set in \mathcal{}, matrices ?
  - should have macros for this.
- small caption font
- Never `\texttt{}` expet it's code.
