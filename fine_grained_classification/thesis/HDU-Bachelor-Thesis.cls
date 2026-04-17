\ProvidesClass{HDU-Bachelor-Thesis}
\NeedsTeXFormat{LaTeX2e}

\DeclareOption{bachelor}{\gdef\@degreetype{本科}}
\DeclareOption{master}{\gdef\@degreetype{硕士}}
\DeclareOption{doctor}{\gdef\@degreetype{博士}}

\ProcessOptions\relax

\newif\if@msthesis
\newif\if@phdthesis
\@msthesisfalse
\@phdthesisfalse

\RequirePackage{iftex}
\RequirePackage{ifxetex}
\RequirePackage{ifluatex}

\newif\ifxeorlua
\ifxeorlua
  \xeorluatrue
\else
  \ifxetex
    \xeorluatrue
  \else
    \ifluatex
      \xeorluatrue
    \else
      \xeorluafalse
    \fi
  \fi
\fi

\LoadClass[a4paper, 12pt, oneside, openany, fontset=none]{ctexbook}

\RequirePackage{amsmath, amssymb}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{array}
\RequirePackage{longtable}
\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{setspace}
\RequirePackage{zhnumber}
\RequirePackage{titlesec}
\RequirePackage{indentfirst}
\RequirePackage[numbers,sort&compress,comma]{natbib}
\RequirePackage{hyperref}
\RequirePackage{pdfpages}

\setmainfont{Times New Roman}
\setsansfont{Arial}
\setmonofont{Courier New}

\ifxeorlua
  \setCJKmainfont{Songti SC}
  \setCJKsansfont{Heiti SC}
  \setCJKmonofont{Songti SC}
  \newCJKfontfamily{\song}{Songti SC}[AutoFakeBold=true]
  \newCJKfontfamily{\hei}{Heiti SC}[AutoFakeBold=true]
  \newCJKfontfamily{\kai}{Songti SC}[AutoFakeBold=true]
\else
  \setCJKmainfont{Songti SC}
  \setCJKsansfont{Heiti SC}
  \setCJKmonofont{Songti SC}
  \newCJKfontfamily{\song}{Songti SC}[AutoFakeBold=true]
  \newCJKfontfamily{\hei}{Heiti SC}[AutoFakeBold=true]
  \newCJKfontfamily{\kai}{Songti SC}[AutoFakeBold=true]
\fi

\geometry{top=2.54cm,bottom=2.54cm,left=3.17cm,right=3.17cm,headheight=1.5cm,footskip=1.75cm}

\parindent 2em
\parskip 0.5ex plus 0.25ex

\setstretch{1.5}

\titleformat{\chapter}[block]{\hei\zihao{3}}{\thechapter}{1em}{}
\titleformat{\section}[block]{\hei\zihao{4}}{\thesection}{1em}{}
\titleformat{\subsection}[block]{\song\zihao{-4}}{\thesubsection}{1em}{}

\ctexset{
  chapter/beforeskip={0pt},
  chapter/afterskip={18pt},
  section/beforeskip={12pt},
  section/afterskip={6pt},
  subsection/beforeskip={6pt},
  subsection/afterskip={6pt}
}

\fancypagestyle{plain}{
  \fancyhf{}
  \fancyfoot[C]{\zihao{-5}\thepage}
  \fancyhead[C]{\zihao{-5}\nouppercase\rightmark}
}

\newenvironment{abstract}{
  \thispagestyle{plain}
  \chapter*{摘要}
  \markboth{摘要}{摘要}
  \addcontentsline{toc}{chapter}{摘要}
}{}

\newenvironment{abstractEN}{
  \thispagestyle{plain}
  \chapter*{Abstract}
  \markboth{Abstract}{Abstract}
  \addcontentsline{toc}{chapter}{Abstract}
}{}

\DeclareRobustCommand\keywords[1]{%
  \par\vbox{\parindent=2em\leftskip=2em\rightskip=2em%
    \hei\noindent\zihao{-4}关键词：#1}
  \par\addtolength{\parskip}{1em}
}

\DeclareRobustCommand\keywordsEN[1]{%
  \par\vbox{\parindent=2em\leftskip=2em\rightskip=2em%
    \textbf{Keywords:} #1}
  \par\addtolength{\parskip}{1em}
}

\endinput
