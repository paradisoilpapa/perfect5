# v229: v228ベース。総合加重単騎評価の直下に、加重2車複全21通り評価表を表示。
# v228: v227ベース。note上部と本文整理から意味不明な「コピー用：xxxx」を完全非表示化。
# v237: v236ベース。加重2車複評価表を旧表に近い全角スペース整形へ戻す。
# v225: v224ベース。2車複本線は◎軸流しではなく、流れ加重的中単騎＋流れ加重妙味単騎から全21通りを再評価し、総合pt上位3点を採用。3連複は従来どおり軸A-BCD-BCDで生成。
# v220: v219ベース。各流れの車番別平均評価（的中順単騎評価）に流れ想定比率を掛けて合算し、2車複サマリーと3連複生成の共通土台にする。
# v221: v220の2車複サマリー改善。流れ加重単騎評価を平均ではなく合算で2車複的中期待へ反映し、本線/抑えが空になる問題を修正。
# v220: v219をベースに、流れ加重単騎評価を3連複だけでなく2車複サマリーにも反映。
# v211: v210をベースに、「イチオシ」を廃止し「ベスト10内重複」へ変更。各流れの総合B以上候補・総合pt上位10内で複数流れに重複した買目を表示。
# v210: v209をベースに、2車複サマリーを固定pt足切りから「総合B以上候補内の順位割合」へ変更。本線=上位30%、抑え=上位50%以内（本線以外）。
# -*- coding: utf-8 -*-
# v207: v206-fixedをベースに、2車複総合ptは√(的中点×妙味点)のまま維持。本線足切りを8.5pt以上へ変更。
# v205: 2車複サマリーのイチオシ/本線に、採用ptが最も高い流れの妙味期待ランクを併記。抑えは従来通りptのみ。
# v203: 会場成績を的中期待/妙味期待の小幅係数へ変換。的中率→的中期待、回収率→妙味期待に反映し、苦手会場で総合B候補が自然に絞られるよう修正。
# v202: 2車複サマリーの全体推奨9pt以上/未満を総合pt降順に並べ替え。流れ別は採用表示を消し、総合B以上候補だけ表示。
# v201: 2車複サマリーの全体推奨を総合pt 9.0以上/9.0未満に分割し、強弱を見やすく整理。
# v200: 冒頭サマリーを1ブロック化。「イチオシ（重複）→全体推奨→流れ別 採用/総合B以上候補」の順に整理して視認性を改善。
# v199: 各流れの総合B以上候補を先に一覧化し、複数流れで重複する買目をイチオシ表示。その後に従来の全体推奨2車複（各流れ上位2点）を表示.
# v197: 2ライン等で渦/逆流が同一主役ラインになる重複を禁止。実ラインが存在しない流れは買目考察から除外。
# v196: 流れ別シナリオの主役ラインを2車複妙味ptへ反映。主役ライン相手を上位保護し、順流/渦/逆流の買目差を強化。
# v195: 順流・渦・逆流の着順予想を、各流域ラインが主役になったシナリオ補正版へ変更。逆流域空欄でも旧逆流ラインを逆流シナリオに補完。
# v194: 買目考察の前に全体推奨2車複サマリー（全体/順流/逆流/渦）を追加。
# v193: 買目考察を順流・逆流・渦の3流れ並列表示へ変更。会場判定good/middle/badによる買目提案を廃止し、各流れの総合B以上・総合pt上位2点を表示。
# v191: 会場判定good系は2車複を出さず、3連複を軸A-候補4車-候補4車の6点型へ変更。middle/bad系はv190維持。
# v198: 2ライン戦などで独立シナリオが成立しない流れは削除せず「該当なし」と表示。
# v192: good系で2車複を出さない場合、「2車複購入候補 該当なし」ブロックを非表示化。
# v191: good系は2車複を出さず、3連複 A-2345-2345 に一本化。
# v190: 反映済み市場印だけで妙味計算。未反映/未入力時の妙味10.0張り付きと、session_state再取得による反映ズレを防止。
# v189: 会場判定middle系は3連複を出さず、2車複を総合評価B以上・総合pt上位4点に切替。good/bad系はv188維持.
# v188: 会場判定middle系は3連複のみ表示（2車複は該当なし）。2車複＋3連複の同時表示はgood系だけに限定。
# v187: 会場判定に応じて購入表示を切替。good系は現行5点、middle系はA-下位4-下位4の三連複6点、bad系は下位4車の2車複BOX6点。
# v186: 競り関与車同士が3連複の軸A・軸Bになる場合、的中1位ではない軸Bを3列目へ降格し、次候補を軸Bへ繰り上げる。
# v180: 開催場決まり手補正の先頭車・番手車補正をv179比50%へ弱化。毎レース表示の買目説明注記5行を削除。
# v179: ライン入力で単騎（1桁ライン）が入力済み車番として認識されない不具合を修正。単騎も1ラインとしてカウントし、入力確認表示を追加。
# v178: オッズパーク等の開催場決まり手成績をサイドバーで数値入力し、1着/2着決まり手率と回数から会場決まり手補正を自動算出。雨天バイアスとは別枠で常時小幅反映.
# v181: 3連複3列目で、軸1・軸2の同ライン残りをスコア選別より先に固定保護。開催場決まり手補正が同ライン保護へ干渉しないよう修正。
# v180: 開催場決まり手補正の先頭・番手補正を50%へ弱化。毎レース表示の長い説明注記を削除。
# v185: 2車複購入候補が1点だけでも3連複を生成する。候補車2車を軸にし、3列目はv181同ライン保護＋補完で常に3車。
# v184: 3連複軸Bを、A以外の2車複候補のうち推奨流れ4位以内で妙味順単騎評価が最も高い車へ変更。
# v177: 3連複候補の軸母集団を「総合評価2車複推奨の2車複購入候補」に変更。軸1は同候補内の的中順最上位、軸2は残りから妙味順位下位側を採用。3列目は推奨流れ側ライン優先で常に3車。
# v176: 3連複候補の1列目・2列目で、妙味順位下位側を選ぶ際に「的中順単騎評価1位」は除外。的中順1位は3列目へ保護する。
# v175: 3連複候補を、B以上候補内の妙味順位下位1位-下位2位を軸にし、的中順1位を3列目へ保護。残り2車はv174の推奨流れ側ライン優先ロジックで常に3車。
# v174: 3連複3列目を「最大3車」ではなく常に3車へ固定。v173の1列目/2列目定義は維持し、不足時は全車候補から補完する。
# v173: 3連複候補の2列目を、的中順2位固定から「総合評価B以上候補に出ている車のうち、1列目を除外して妙味順位下位側の車」へ変更。3列目ロジックはv169を維持。
# v172: 2車複購入候補を「総合評価B以上の総合pt上位2点のみ」に固定。妙味A+/pt差による条件付き3点目追加を廃止。3連複候補の土台は従来どおり総合評価B以上候補。
# v171: 2車複購入候補を「総合評価B以上の総合pt上位2点＋条件付き3点目（妙味A+以上、または2点目との差0.3以内）」へ修正。総合評価B上位を妙味A+縛りで落とさない。
# v170: 2車複購入候補を「総合評価B以上 かつ 妙味期待A+以上」に絞る。3連複候補の軸・3列目生成は従来どおり総合評価B以上候補を土台にして本線を拾う。
# v169: 3連複3列目は採用された推奨流れの流域ラインを最優先。軸は従来通り的中順単騎評価で選び、3列目は推奨流れ側ラインのヒモ→突っ込む別線側の直近1車→B以上残り→妙味補助の順で最大3車に絞る。
# v168: 3連複購入候補の3列目を、B以上残りの単純スライドから、軸2車のライン直近相手・推奨流れ上位・妙味順単騎評価を加点して最大3車まで再選別する方式へ変更。
# v167: 推奨流れをKO上位3車の流域多数決で補正。順流/渦/逆流の所属が2車以上ならその流れ、3車が割れた場合は逆流扱い。H主導寄せより後で適用。
# v166: 【２車複考察】を【買目考察】へ変更。総合評価B以上の2車複候補から車番を抽出し、的中順単騎評価上位2車を軸、残りを3列目にした3連複購入候補を追加。
# v165: 2車複候補を総合評価B以上のpt順表示へ戻し、車番別平均評価の結論順1:2/1:3を非表示。的中順/妙味順を単騎評価表記へ変更。
# v108: note上部サマリーに「2車複｜妙味通過（7.0pt以上）」だけを復活。評価重複・三連複妙味・三連複評価重複はnote上部へ出さない。
# v111: 選択コピー欄の2車複妙味通過表示を簡潔化。旧妙味通過＋34-12内通過ペアを統合し、説明文は表示しない。基準8.5pt。
# v114: note上部推奨を二強軸フォメ＋安め上位4点表記へ変更。補助2車複は妙味8.5pt通過のみを短く表示。
# v120: 全体妙味A/B/C変換の二重適用を修正。旧ラベルは表示直前に一度だけ変換し、青網掛けとコピー欄を一致させる。
# v163: note用コピーエリア上部の青網掛け情報ボックス（推奨戦法・全体妙味/旧フォメ表示）を非表示化。
# v164: ライン評価グループで順流域は代表1ラインだけに制限。単騎も1ラインとして扱い、余剰順流ラインは渦域/逆流域へ再配分。
# v161: 車番別平均評価（極端値除外）を上下1本ずつ除外したトリム平均に変更。結論順1:2/1:3はトリム平均で計算。
# v162: 棚卸版。表示していない旧3連複まとめ候補/3列目候補の計算ブロックを削除し、現行中核を2車複BOX評価＋車番別トリム平均評価へ整理。
# v160: 車番別平均評価（極端値除外）を着順率係数なしの素平均に戻し、結論順を 的中:妙味=1:2 / 1:3 の2系統で表示。
# v157: 車番別平均評価（極端値除外）の妙味順係数を2着率から3着内率へ変更。3連複まとめ候補・3列目候補ブロックを非表示化。
# v155: 車番別平均評価（極端値除外）を再設計。的中順は1着率係数、妙味順は2着率係数を掛け、結論順はその平均で並べる。
# v154: 車番別平均評価（極端値除外）に「結論順」を追加。的中平均×妙味軸平均で、着順確率を加味した最終順位を表示する。
# v121: note上部推奨を三連複固定表示からステップ式（1-2幹確認→123BOX→1/2軸拡張）へ変更。
# v122: A-B同一ライン時、B後ろの3番手以降をライン残り候補としてステップ3に保護。地区まとめは弱めるが即消ししない。
# v122: コメントチェックに自在・競り相手・3番手以降追走信頼を追加。競り相手同士の弱者追加減点、3番手以降の結束補正、KO差＋競り＋脚質による1軸/二強/混戦判定をnote上部ステップ式へ反映。
# v123: note上部の補助候補表示を「長期スパン妙味｜12-34」へ変更。20倍以上のみ候補として1-3/1-4/2-3/2-4相当をpt付きで固定表示。
# v127: 長期スパン妙味2車複の評価5位参照E未定義を修正。評価1・2×評価2〜5フォーメーション生成時にEを安全定義。
# v128: note上部サマリーからステップ式ブロックを削除。軸判定と長期スパン妙味2車複だけを表示。
# v130: 長期スパン妙味2車複を、的中期待・妙味期待・総合評価A/B/C/Dの2軸表示へ変更。
# v131: 長期スパン妙味2車複の購入目安を20倍以上から総合B以上へ変更。点数過多時はA優先。
# v134: 的中期待の計算を掛け算から、打ち合わせ通り 0.6×VeloBi点 + 0.4×Win点 + 一致ボーナスへ修正。
# v136: 2車単候補条件を「的中期待Aかつ総合C/D」へ拡張。
# v143: 妙味A++/A+/Aの基準を厳格化し、買い目表を区切り線形式に変更して表示崩れを軽減。
# v142: 妙味期待を妙味ptだけでA++/A+/A/B/C/Dに細分化。総合評価はA++/A+/Aを妙味A扱いに正規化。
# v137: 2車単候補を廃止。2車複は総合pt上位2点（微差なら3点）に絞り、3連複まとめ候補を追加。3列目は評価別3着内率＋ライン/展開/妙味補正で7車全体から再計算。
# v138: 3連複まとめ候補の3列目を、3列目pt上位2車までに制限。買い目増加で合成オッズを下げすぎないため。
# v139: 2車複候補をフォーメーション固定から全車BOX（全21通り）総合pt順へ変更。表示も全候補を総合pt順に出す。
# v135: 総合Cかつ的中期待Aの買い目を、2車複ではなく2車単候補として別表示。
# v133: ２車複フォーメーションに総合評価別の買い目まとめ（A/B/C/D）を追加。C表記を「やや見送り」へ変更。
# v132: 長期スパン妙味2車複の見出しを2車複フォーメーションへ整理。Aを推奨買い候補へ変更し、C/Dも20倍以上なら買い推奨の注記を追加。
# v129: ステップ式削除後に残っていた軸判定の「上限：ステップ◯まで」表示を削除。
# v126: 長期スパン妙味2車複を締切3分前13倍以上推奨へ変更。評価1・2 × 評価2〜5のフォーメーション表示へ拡張。
# v124: 後位信頼をselectboxからチェックボックス式へ変更。複数チェック時は単騎寄り＞流動＞地区まとめ＞明確追走の優先順で1つの内部ラベルに変換。
# v125: コメントチェックを自力/自力自在/自在に分離。単騎コメントを後位信頼から分離して独立チェック化。後位信頼は明確/地区/流動のみ。
# v113: 三連複推奨の買い基準文言のみ削除。余計な代替文言は出さない。
# v116: 三連複二強軸フォメの3列目を車番羅列ではなく「全」表示へ修正。
# v117: note三連複推奨を評価1・2-全-全表示へ変更。2列目3車固定を廃止し、安め上位4点でライン決着も拾える表示へ修正。
# v107: 本文条件を1-2市場ワイドオッズ表示へ修正。払戻合計入力と推定オッズ表示を削除し、三連複/34-12必要合成オッズを本文へ表示。
# v106: 1-2市場2車複条件を推奨下限合成オッズから切り離し、1-2二車複的中分布の推定想定オッズを表示。
# v105: note三連複推奨の買い基準にサイドバー計算の推奨下限合成オッズを本文差し込み。
# v104: 1-2市場2車複の条件を固定3倍ではなく、推奨下限合成オッズから表示。3倍固定文言を削除。
# v103: note推奨にサイドバー入力の意味を反映。1-2ワイド率・推奨下限・1-2二車複基準以下率を短く表示。
# v100: note三連複推奨を実戦用短文へ整理。1-2ワイド率だけで下限計算、1-2市場2車複3倍以下対象R数をサイドバー入力へ追加。
# v97: 推奨流れ1-2/12-34系の集計値は未入力時に規定値計算しない。サイドバー入力後のみ判定基準を算出。
# v96: 推奨流れ1-2-全三連複/34-12二車複の切替基準を固定2.5倍から集計入力ベースの自動計算へ変更。
# v95: note最終推奨の切替条件にも34-12 2車複フォメの実車番を必ず表示。
# v94: noteコピー欄を最終推奨中心へ圧縮。列評価・旧フォメ・会場H詳細ログをnote出力から除外。
# v93: v92の未入力ステータス表示を削除。合成実効オッズ未入力時は基本推奨だけを表示し、余計な状態文を出さない。
# v92: 基本推奨に評価順1-2-全 三連複を追加。合成実効オッズ2.5倍未満の時だけ34-12 2車複フォメを切替推奨表示。
# v99: note推奨を短文化。不要な合成実効オッズ入力・12-34確率入力を削除し、1-2ワイド確率だけで下限表示。
# v98: note最終推奨の説明文を短縮。判定基準は的中率と推奨下限合成オッズのみ表示。
# v90: 三展フォメを廃止し、推奨流れ34-12二車複フォメをメイン表示とnoteコピー両方へ出す。
# v91: 推奨流れ34-12フォメの説明行（順位対応/買う/切る）を削除し、買い目だけを簡潔表示。
# v90: 推奨流れ34-12をメイン表示とコピー欄へ出力。
# v85: 会場判定・最終H補正倍率・必要オッズ倍率の3点を三展+KO順位生成へ反映。悪い会場ほどKO/H補正順位を強く見る。
# v84: 必要オッズ倍率を三展+KO順位へ反映。倍率が高い会場ほど三展固定を弱め、KO/H補正順位を強く見る。
# v83: 三展開合成フォメ上部ブロックを必ず三展+KOスコア順位ベースの1券種1行表示に統一。生成不可フォールバックを廃止。
# v82: 三展開合成フォメを1券種1行表示へ修正。展開・抑え2車単の旧まとめ表示を廃止。
# v81: 三展開合成フォメを評価123・安め切りBOX型（1→2→3三連単＋2→1/3→1二車単＋1=3/2=3二車複）へ変更。
# v80: 会場別の的中率/回収率手入力→最終H1番手減点・2番手ライン加点補正を買い目用スコアへ反映。
# v79: v78でreturnに nitan_forme/nitan_follow を渡しておらず表示されない不具合を修正。
# v78: 抑え2車単 23→1 を三展開合成フォメ直下へインライン表示（例：抑え2車単：54→7）。
# v77: 三展開合成フォメ 1-23-24 の抑えとして、2車単 23→1（例：54→7）を「抑え2車単」で表示。
# v69: 素材4列表示で、4列目分離前の3列目候補を保持。軸ライン直近相手を4列目から復帰し、弱い別線・末端を4列目へ回す。
# v68: 素材表示4列化の3列目圧縮で、軸ライン直近相手を優先保護。
#      妙味ptだけで長い軸ライン末端を残しすぎず、弱い・深い候補を4列目へ回す。
# v70: 素材表示の2列目を信頼2車へ圧縮。妙味高pt車は2列目でなく3列目へ回し、弱線・薄線を4列目へ分離。
# v71: 素材表示の3列目を修正。弱い単騎妙味より、ライン持ち妙味＋軸ライン直近相手を優先する。
# v76: 三展+KOは三連単 1-23-24 の3点へ戻し、ズレ保険として2車単 23→1 を同時出力。
# v75: 三展スコアにKO実スコアを加算。三展開合成フォメは 1-23-245 型、原則5点へ拡張。
# v74: 三展スコア順位を追加。三展開合成フォメは三展スコア順位から 1-23-24 型で生成し、VeloBi列評価は素材として維持。
# v73: 三展開合成フォメの3列目コピー補正を修正。妙味残りより軸ライン直近相手を優先し、三展開で薄い単騎妙味を3列目へ入れない。
# v72: 三展開合成フォメの3列目が2列目コピーになる場合、妙味残り＋軸ライン直近相手で再構成する。
# v67: 4列目は「素材表示だけ」に限定。三展開合成フォメ・妙味通過・期待値推奨へ副作用を出さない。
#      PILLAR_EXCLUDE_THIRD_CARS を無効化し、三連複フォメ表示用だけ col3 を圧縮する.
# v64: 三展開合成フォメを最終購入3点へ圧縮。素材フォメは維持し、A-BC-CD型で攻守バランスを取る。
# v35: 評価重複のみの場合、2列目は低pt重複2車複の2セットまで。残り重複と軸ライン残りを3列目へ回す
# v37: 評価重複2車複が1セットのみなら、軸ライン残りが基本2列目にある場合は2列目にも追加する
# v52: 妙味2車複が複数ある時、軸ライン残りを無条件で3列目へ入れず、採用2着候補のライン残り＋評価重複相手だけを3列目へ回す。
# v53: 妙味2車複が1点だけの場合、評価重複を足す前に「軸ライン相手」が基本2列目にあれば2列目へ優先採用する。
# v39: 三連複柱ありで2列目を低pt2セットに絞る場合、2セット目以降のライン残りも3列目へ補正。全候補が基本3列目内なら基本3列目順を優先。
# v54: ライン補正フォメの3列目は、採用2着候補の同ライン全員ではなく「直近の相手」まで。ライン末尾・4番手格は3列目から落とす。
# v56: 4列目の定義を修正。単なるライン末尾ではなく、4車ライン、または軸ではない3車以上ラインの「ライン内VeloBi評価3番手以降」を4列目へ分離する。
# v57: 4列目へ落とす条件を再修正。2列目採用車・軸との2車複妙味通過車は、長いラインの3番手以降でも3列目に残す。
# v58: 4列目条件を再修正。2列目にいるだけでは保護しない。長いラインの評価3番手以降は原則4列目、ただし「2列目採用かつ軸との妙味通過」は3列目に残す。
# v59: 推奨ライン補正フォメでも4列目候補を3列目へ戻さない。基本フォメで4列目に落とした車は、ライン補正のextras/third_seedから除外する。
# v60: 妙味2車複が複数ある時も、軸ラインの直近相手は3列目へ残す。ライン相手を消して補正フォメが崩れるのを防ぐ。
# v61: 妙味2車複が複数ある時、軸ライン相手が基本2列目にいるなら2列目へ優先採用し、押し出された妙味相手は3列目へ回す。
# v62: 補正フォメを固定仕様へ整理。2列目最大2車、2車ラインの軸相手のみ強制優先、3列目は採用2着候補の直近非軸ライン相手・押し出し妙味・評価重複残りだけ。4列目は戻さない。
# v63: v62で推奨フォメブロックが消える不具合を修正。妙味起点時にA/B/C・ライン情報・直近ライン相手関数を先に初期化してから補正フォメ生成する。
# v42: 基本三連複フォメの3列目で、2列目採用車の同ライン残りを必ず残す（例：5を2列目なら52の2を3列目へ）。
# v44: 三連複妙味ptをVeloBi順位寄りに再調整。外部印ズレの10点張り付きと同一三連複の重複表示を抑制。
# v45: 三連複妙味ptで軸の市場印を上限キャップ化。評価1が△/〇/◎なら10点張り付きさせない。
# v46: 2車複妙味ptにも軸の市場印キャップを適用。軸が△/〇/◎なら2車複も10点張り付きさせない。
# v49: v46〜v48のキャップが強すぎたため撤廃。市場印は減点として反映し、VeloBi筋の妙味は残す。
# v50: 2車複妙味ptの市場印取得と相手印評価を修正。軸印より相手印の濃淡を強く反映し、◎軸×無印相手と◎軸×△相手を同点にしない。
# v47: 市場印snapshotが—入りでfallbackされない問題を修正。2車複の軸印キャップも通過基準未満へ強化。
# v48: snapshotだけでなく現在のst.session_state上の車番別市場印も後段で再取得し、2車複ptへ確実に反映。
# v140: 2車複BOX評価のB/C/Dランク別まとめ行を削除。全21通りの買い目表だけを総合pt順で表示する。
# v141: 全21通りの買い目表と2車複購入候補の並びを、総合評価ランク優先ではなく総合pt降順へ修正。
# v148: 買い目表を縦線なしのまま、全角スペース主体の固定幅に変更。日本語見出しとA/A+/A++の見た目を揃える。
# v149: 買い目表の列開始位置を固定幅10に統一。縦線なしで、見出し位置に合わせて各列を整列。
# v150: 買い目表を「見出しは左寄せ、A/A+/A++等のランクは列内中央寄せ」に変更。縦線なしのまま視認性を改善。
# v152: 全21通りの2車複内部数値から車番別平均（的中期待順・妙味順・総合順）を追加表示。
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata, re
import math, json, requests
from statistics import mean, pstdev
from itertools import combinations
from datetime import datetime, date, time, timedelta, timezone

def _grep_self(pattern: str, path: str = __file__, context: int = 2):
    """
    grep -n の代わり：このファイル(path)を読み、patternを含む行番号を出す
    context: 前後に何行表示するか
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

    hits = []
    for i, line in enumerate(lines, 1):
        if pattern in line:
            hits.append(i)

    if not hits:
        print(f"[SELF-GREP] not found: {pattern!r} in {path}")
        return []

    print(f"[SELF-GREP] found {len(hits)} hit(s): {hits}  pattern={pattern!r}")
    for ln in hits:
        s = max(1, ln - context)
        e = min(len(lines), ln + context)
        print("-----")
        for j in range(s, e + 1):
            mark = ">>" if j == ln else "  "
            print(f"{mark}{j:5d}: {lines[j-1].rstrip()}")
    return hits



# ==============================
# 偏差値T（車番→T）自動検出ユーティリティ
# ==============================
def _extract_car_t_map_from_obj(obj):
    """
    obj から「車番→偏差値T(dict)」を取り出す。
    - dict: {1: 52.3, "4": 47.1, ...}
    - Series: indexが車番
    - 1列DataFrame: indexが車番
    """
    if obj is None:
        return None

    # dict
    if isinstance(obj, dict) and obj:
        out = {}
        for k, v in obj.items():
            ks = "".join(ch for ch in str(k) if ch.isdigit())
            if not ks:
                continue
            try:
                out[ks] = 50.0 if v is None else float(v)
            except Exception:
                continue
        return out if out else None

    # pandas Series
    if isinstance(obj, pd.Series) and not obj.empty:
        out = {}
        for k, v in obj.to_dict().items():
            ks = "".join(ch for ch in str(k) if ch.isdigit())
            if not ks:
                continue
            try:
                out[ks] = 50.0 if v is None else float(v)
            except Exception:
                continue
        return out if out else None

    # pandas DataFrame（1列だけ偏差値が入ってる想定）
    if isinstance(obj, pd.DataFrame) and (not obj.empty):
        if obj.shape[1] >= 1:
            s = obj.iloc[:, 0]
            return _extract_car_t_map_from_obj(s)

    return None


def _looks_like_t_map(tmap, active_cars=None):
    if not isinstance(tmap, dict) or not tmap:
        return False

    keys = [k for k in tmap.keys() if str(k).isdigit()]
    if len(keys) < 4:
        return False

    vals = []
    for k in keys:
        try:
            vals.append(float(tmap[k]))
        except Exception:
            pass

    if len(vals) < 4:
        return False

    in_range = [v for v in vals if 10.0 <= v <= 90.0]
    if len(in_range) / len(vals) < 0.8:
        return False

    m = sum(in_range) / len(in_range)
    if not (25.0 <= m <= 75.0):
        return False

    if active_cars:
        ac = [str(x) for x in active_cars if str(x).isdigit()]
        if ac:
            hit = sum(1 for x in ac if x in tmap)
            if hit / len(ac) < 0.6:
                return False

    return True


def _pick_hensachi_source_from_globals(g, active_cars=None):
    """
    globals() から偏差値Tソースを自動選別して (tmap, name, score) を返す
    """
    best = None
    best_name = None
    best_score = -1.0

    for name, obj in g.items():
        if name.startswith("__"):
            continue
        tmap = _extract_car_t_map_from_obj(obj)
        if not tmap:
            continue
        if not _looks_like_t_map(tmap, active_cars=active_cars):
            continue

        ac = [str(x) for x in (active_cars or []) if str(x).isdigit()]
        hit = sum(1 for x in ac if x in tmap) if ac else len(tmap)
        coverage = (hit / len(ac)) if ac else 0.5

        vals = [float(v) for v in tmap.values() if isinstance(v, (int, float))]
        uniq = len(set(round(v, 2) for v in vals)) / max(1, len(vals))

        score = coverage * 0.7 + uniq * 0.3

        if score > best_score:
            best_score = score
            best = tmap
            best_name = name

    return best, best_name, best_score


# =========================================================
# 必須：グローバル共通部品（参照より先に必ず定義）
# =========================================================

def _digits_of_line(ln):
    s = "".join(ch for ch in str(ln) if ch.isdigit())
    return [int(ch) for ch in s] if s else []

# _PATTERNS をどこかで for で回しているなら、最低限ここで存在させる
_PATTERNS = []   # ← まず NameError を止めるための保険（本来は下で登録する）





# =========================================================
# v162 棚卸メモ（現行で使う中核）
# 1) 推奨流れ・軸判定
# 2) 2車複：全21通りBOX評価 → 総合pt上位2点（微差なら3点）
# 3) 車番別平均評価：各車を含む2車複6通りから最高/最低を除外したトリム平均
# 4) note表示：2車複考察・車番別平均評価・全21通り表
#
# 非表示/旧仕様として今回削ったもの：
# ・旧3連複まとめ候補の生成
# ・旧3列目候補の専用pt生成
# ※三展開・旧フォメ系の関数は周辺計算との依存が残る可能性があるため、今回は削除せず表示から隔離。
# =========================================================

# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="ヴェロビ：級別×日程ダイナミクス（5〜9車・買い目付き / 統合版）", layout="wide")

# ==============================
# ★ 新規パラメータ（偏差値＆推奨ロジック）
# ==============================
HEN_W_SB   = 0.20   # SB重み
HEN_W_PROF = 0.30   # 脚質重み
HEN_W_IN   = 0.50   # 入着重み（縮約3着内率）
HEN_DEC_PLACES = 1  # 偏差値 小数一桁

HEN_THRESHOLD = 55.0     # 偏差値クリア閾値
HEN_STRONG_ONE = 60.0    # 単独強者の目安

MAX_TICKETS = 6          # 買い目最大点数

# 推奨流れ1-2-全 三連複 → 推奨流れ34-12 2車複フォメ切替基準
# v97: 集計値はサイドバー入力後だけ使う。未入力時に規定値（92/45/31等）で自動計算しない。
FLOW_SWITCH_DEFAULT_TOTAL_RACES = None
FLOW_SWITCH_DEFAULT_12_WIDE_HITS = None
FLOW_SWITCH_DEFAULT_12_NIFUKU_HITS = None
FLOW_SWITCH_DEFAULT_12_NIFUKU_UNDER3_RACES = None
FLOW_SWITCH_DEFAULT_TARGET_EV = None
FLOW_SWITCH_DEFAULT_SAFETY = None
FLOW_12_ALL_TRIO_SWITCH_ODDS_THRESHOLD = None

def _safe_int_or_none(v):
    try:
        if v is None or v == "":
            return None
        return int(v)
    except Exception:
        return None


def _safe_float_or_none(v):
    try:
        if v is None or v == "":
            return None
        x = float(v)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None




# ==============================
# v178：開催場決まり手成績 → 会場決まり手補正
# ==============================
# 入力値はオッズパーク等の表をそのまま転記する想定。
# 補正量はコード側で作るため、後から係数だけ調整できる。

VENUE_KIMARITE_BASELINE = {
    "win_escape": 25.0,
    "win_sashi": 50.0,
    "win_makuri": 25.0,
    "sec_escape": 20.0,
    "sec_sashi": 30.0,
    "sec_makuri": 15.0,
    "sec_mark": 35.0,
}

def _pct_input_to_float(v, default=0.0):
    """13.9 / 13.9% / 0.139 のどれでも受ける。戻り値は％値。"""
    try:
        if v is None or v == "":
            return float(default)
        x = float(str(v).replace("%", "").strip())
        if 0.0 < x <= 1.0:
            x *= 100.0
        if not math.isfinite(x):
            return float(default)
        return float(clamp(x, 0.0, 100.0))
    except Exception:
        return float(default)

def _venue_kimarite_reliability(sample_count):
    """回数が少ない時は補正を弱める。150回以上は満額。"""
    try:
        n = int(float(sample_count or 0))
    except Exception:
        n = 0
    if n <= 0:
        return 0.0
    return float(clamp((n / 150.0) ** 0.5, 0.35, 1.0))

def _calc_venue_kimarite_role_bonus_map(stats, max_abs=0.35):
    """
    開催場決まり手成績から、ライン役割別の小幅補正を作る。
    head      : 逃げ/先行残り
    second    : 番手差し・2着マーク
    thirdplus : 後位マーク残り
    single    : 捲り/単騎一撃
    """
    if not isinstance(stats, dict) or not stats.get("enabled", False):
        return {"head":0.0, "second":0.0, "thirdplus":0.0, "single":0.0}, 0.0, {}

    base = VENUE_KIMARITE_BASELINE
    rel = _venue_kimarite_reliability(stats.get("sample_count", 0))
    if rel <= 0.0:
        return {"head":0.0, "second":0.0, "thirdplus":0.0, "single":0.0}, 0.0, {}

    d = {
        k: _pct_input_to_float(stats.get(k, base[k]), base[k]) - base[k]
        for k in base.keys()
    }

    # 1%差を何ptに変換するか。
    # v180: 先頭車・番手車の補正だけv179比50%へ弱化。
    # v181: 3列目保護へ干渉させないため、thirdplus / single への会場決まり手補正は使わない。
    raw = {
        "head":      0.010*d["win_escape"] + 0.005*d["sec_escape"],
        "second":    0.010*d["win_sashi"]  + 0.005*d["sec_mark"] + 0.003*d["sec_sashi"],
        "thirdplus": 0.0,
        "single":    0.0,
    }

    role_bonus = {
        k: float(clamp(v * rel, -float(max_abs), float(max_abs)))
        for k, v in raw.items()
    }

    detail = {"diff": d, "raw": raw, "reliability": rel, "max_abs": float(max_abs)}
    return role_bonus, rel, detail

def _fmt_signed_pt(v):
    try:
        return f"{float(v):+.2f}pt"
    except Exception:
        return "+0.00pt"

def _apply_venue_kimarite_to_score_map(score_map, line_def, stats):
    """score_mapへ会場決まり手補正を常時小幅反映する。"""
    role_bonus, rel, detail = _calc_venue_kimarite_role_bonus_map(stats)
    reason_map = {}
    out = dict(score_map or {})

    for k in list(out.keys()):
        try:
            car = int(k)
            role = role_in_line(car, line_def) if isinstance(line_def, dict) else "single"
            if role not in role_bonus:
                role = "single"
            b = float(role_bonus.get(role, 0.0) or 0.0)
            out[k] = float(out.get(k, 0.0) or 0.0) + b
            reason_map[car] = f"{role}:{_fmt_signed_pt(b)}"
        except Exception:
            continue

    return out, role_bonus, rel, detail, reason_map

def _safe_div_float(a, b, default=None):
    try:
        a = float(a)
        b = float(b)
        if b <= 0:
            return default
        return a / b
    except Exception:
        return default


def _calc_flow_switch_metric(hit_count, total_count, target_ev, safety_factor):
    hit_rate = _safe_div_float(hit_count, total_count, None)
    target_ev = _safe_float_or_none(target_ev)
    safety_factor = _safe_float_or_none(safety_factor)

    out = {
        "hit_rate": hit_rate if hit_rate and hit_rate > 0 else None,
        "break_even_odds": None,
        "ev_required_odds": None,
        "recommended_floor_odds": None,
    }
    if out["hit_rate"] is None:
        return out

    out["break_even_odds"] = 1.00 / out["hit_rate"]
    if target_ev is not None and target_ev > 0:
        out["ev_required_odds"] = target_ev / out["hit_rate"]
    if out["ev_required_odds"] is not None and safety_factor is not None and safety_factor > 0:
        out["recommended_floor_odds"] = out["ev_required_odds"] * safety_factor
    return out


def _calc_flow_switch_stats(total_count, wide12_hits, target_ev=None, safety_factor=None,
                            nifuku12_under3_races=None, nifuku12_hit_count=None):
    """
    推奨流れ1-2-全 三連複の下限計算用。

    分けるもの：
    ・本文条件：1-2市場ワイドオッズ 2.04倍以下 / 2.05倍以上。
      ※現行集計の切替目安を本文に出す。ここに推奨下限合成オッズを流用しない。
    ・三連複買い基準：推奨1-2ワイド的中率から推奨下限合成オッズを出す。
    ・34-12買い基準：34-12二車複フォメ的中率から必要合成オッズを出す。

    払戻合計入力や、3倍以下ゾーンの推定平均オッズは使わない。
    """
    total_i = _safe_int_or_none(total_count)
    wide_i = _safe_int_or_none(wide12_hits)
    target_f = _safe_float_or_none(target_ev)
    safety_f = _safe_float_or_none(safety_factor)
    hit_i = _safe_int_or_none(nifuku12_hit_count)
    under3_i = _safe_int_or_none(nifuku12_under3_races)

    # 現行集計で確認した切替目安。本文表示用。
    # 2.04以下 → 三連複1-2-全候補、2.05以上 → 34-12 2車複切替候補。
    wide_market_switch_odds = 2.04

    # 現行集計の34-12的中数：1-3=12、1-4=7、2-3=7、2-4=5、合計31。
    # 34-12フォメ必要合成オッズは総レース数と同じEV/安全係数で算出する。
    switch_3412_hits = 31 if total_i and total_i > 0 else None

    return {
        "total_count": total_i,
        "wide12_hits": wide_i,
        "nifuku12_hit_count": hit_i,
        "nifuku12_under3_races": under3_i,
        "nifuku12_under3_rate": _safe_div_float(under3_i, hit_i, None),
        "wide_market_switch_odds": wide_market_switch_odds,
        "target_ev": target_f,
        "safety_factor": safety_f,
        "trio12_all": _calc_flow_switch_metric(wide_i, total_i, target_f, safety_f),
        "switch_3412_hits": switch_3412_hits,
        "switch_3412": _calc_flow_switch_metric(switch_3412_hits, total_i, target_f, safety_f),
    }

def _fmt_pct(v):
    try:
        if v is None:
            return "—"
        return f"{float(v) * 100:.1f}%"
    except Exception:
        return "—"

def _fmt_odds(v):
    try:
        if v is None:
            return "—"
        return f"約{float(v):.2f}倍"
    except Exception:
        return "—"


def _fmt_plain_odds(v):
    try:
        if v is None:
            return "—"
        x = float(v)
        if abs(x - round(x)) < 1e-9:
            return f"{x:.0f}倍"
        return f"{x:.1f}倍"
    except Exception:
        return "—"




def _fmt_count_rate_line(hit_count, total_count, rate):
    try:
        h = _safe_int_or_none(hit_count)
        t = _safe_int_or_none(total_count)
        if h is None or t is None or t <= 0 or rate is None:
            return "—"
        return f"{h} / {t} = {_fmt_pct(rate)}"
    except Exception:
        return "—"

def _flow12_market_wide_basis_odds(stats=None):
    """本文条件に使う1-2市場ワイドオッズ目安。
    推奨下限合成オッズや払戻推定は使わない。
    """
    try:
        if stats is None:
            stats = globals().get("FLOW_SWITCH_STATS", None) or _get_flow_switch_stats_from_state()
        v = (stats or {}).get("wide_market_switch_odds", None)
        if v is None:
            return None
        v = float(v)
        return v if math.isfinite(v) and v > 0 else None
    except Exception:
        return None


def _flow12_market_wide_condition_lines(inline_switch=False, stats=None):
    v = _flow12_market_wide_basis_odds(stats)
    if v is None:
        if inline_switch:
            return "1-2市場ワイドオッズ 目安超"
        return "1-2市場ワイドオッズ 目安以下"
    if inline_switch:
        return f"1-2市場ワイドオッズ {v + 0.01:.2f}倍以上"
    return f"1-2市場ワイドオッズ {v:.2f}倍以下"

# 旧名互換：既存呼び出しは残し、中身だけワイド条件へ差し替える。
def _flow12_market_nifuku_basis_odds(stats=None):
    return _flow12_market_wide_basis_odds(stats)


def _flow12_market_nifuku_basis_label(stats=None):
    v = _flow12_market_wide_basis_odds(stats)
    if v is None:
        return "目安"
    return f"{v:.2f}倍"


def _flow12_market_nifuku_condition_lines(inline_switch=False, stats=None):
    return _flow12_market_wide_condition_lines(inline_switch, stats)


def _flow12_trio_buy_criteria_line(stats=None):
    """ヴェロビ三連複推奨の運用目安表示。

    現行運用は「安目切り後の合成オッズで買う/見送る」ではなく、
    市場の安め上位を進塁目として使うため、旧合成オッズ基準は本文に出さない。
    """
    return "安め上位4点セットを基本"



def _flow3412_nifuku_buy_criteria_line(stats=None):
    """34-12二車複フォメの必要合成オッズ表示。"""
    try:
        if stats is None:
            stats = globals().get("FLOW_SWITCH_STATS", None) or _get_flow_switch_stats_from_state()
        sw = (stats or {}).get("switch_3412", {}) or {}
        floor = sw.get("recommended_floor_odds", None)
        if floor is None:
            return "34-12二車複フォメ合成オッズが推奨下限以上"
        floor = float(floor)
        if not math.isfinite(floor) or floor <= 0:
            return "34-12二車複フォメ合成オッズが推奨下限以上"
        return f"34-12二車複フォメ合成オッズ {floor:.2f}倍以上"
    except Exception:
        return "34-12二車複フォメ合成オッズが推奨下限以上"

def _fmt_ev_required_label(target_ev):
    try:
        if target_ev is None:
            return "EV必要合成オッズ："
        return f"EV{float(target_ev):.2f}必要合成オッズ："
    except Exception:
        return "EV必要合成オッズ："

def _get_flow_switch_stats_from_state():
    total = st.session_state.get("flow_switch_total_races", None)
    nifuku_hits = st.session_state.get("flow_switch_12_nifuku_hits", None)
    under3_races = st.session_state.get("flow_switch_12_nifuku_under3_races", None)
    wide_hits = st.session_state.get("flow_switch_12_wide_hits", None)
    target_ev = st.session_state.get("flow_switch_target_ev", None)
    safety = st.session_state.get("flow_switch_safety_factor", None)
    return _calc_flow_switch_stats(total, wide_hits, target_ev, safety, under3_races, nifuku_hits)


def _flow_12_all_recommended_floor():
    stats = globals().get("FLOW_SWITCH_STATS", None)
    if not stats:
        try:
            stats = _get_flow_switch_stats_from_state()
        except Exception:
            stats = None
    try:
        floor = stats.get("trio12_all", {}).get("recommended_floor_odds", None) if stats else None
        if floor is not None and float(floor) > 0:
            return float(floor)
    except Exception:
        pass
    return None

def _make_flow_switch_pairs(xs):
    if len(xs) < 4:
        return []
    A, B, C, D = int(xs[0]), int(xs[1]), int(xs[2]), int(xs[3])
    raw_pairs = [(C, A), (C, B), (D, A), (D, B)]
    pairs = []
    keys = set()
    for a, b in raw_pairs:
        if int(a) == int(b):
            continue
        key = tuple(sorted((int(a), int(b))))
        if key in keys:
            continue
        keys.add(key)
        pairs.append((int(a), int(b)))
    return pairs

def _append_flow_switch_criteria_lines(lines, stats, include_headers=True):
    total = int(stats.get("total_count") or 0)
    wide_hits = int(stats.get("wide12_hits") or 0)
    trio = stats.get("trio12_all", {}) or {}

    if include_headers:
        lines.append("【1-2-全 三連複 判定基準】")
        lines.append("")
    lines.append("現在の推奨流れ1-2ワイド的中率：")
    if total > 0:
        lines.append(f"{wide_hits} / {total} = {_fmt_pct(trio.get('hit_rate'))}")
    else:
        lines.append("—")
    lines.append("")
    lines.append("推奨下限合成オッズ：")
    lines.append(_fmt_odds(trio.get("recommended_floor_odds")))



# 推奨ラベル判定用（クリア台数→方針）
# k>=5:「2車複・ワイド」中心（広く） / k=3,4:「3連複」 / k=1,2:「状況次第（軸流し寄り）」 / k=0:ケン
LABEL_MAP = {
    "wide_qn": lambda k: k >= 5,
    "trio":    lambda k: 3 <= k <= 4,
    "axis":    lambda k: k in (1,2),
    "ken":     lambda k: k == 0,
}

# 期待値レンジ（内部基準で使用可。画面非表示）
P_FLOOR = {"sanpuku": 0.06, "nifuku": 0.12, "wide": 0.25, "nitan": 0.07, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60

# ==============================
# 既存：風・会場・マスタ
# ==============================
WIND_COEFF = {
    "左上": -0.03, "上": -0.05, "右上": -0.035,
    "左": +0.05,  "右": -0.05,
    "左下": +0.035, "下": +0.05, "右下": +0.035,
    "無風": 0.0
}
WIND_MODE = "speed_only"
WIND_SIGN = -1
WIND_GAIN = 3.0
WIND_CAP  = 0.10
WIND_ZERO = 1.5
SPECIAL_DIRECTIONAL_VELODROMES = {"弥彦", "前橋"}

SESSION_HOUR = {"モーニング": 8, "デイ": 11, "ナイター": 18, "ミッドナイト": 22}
JST = timezone(timedelta(hours=9))

BASE_BY_KAKU = {"逃":1.58, "捲":1.65, "差":1.79, "マ":1.45}

KEIRIN_DATA = {
    "函館":{"bank_angle":30.6,"straight_length":51.3,"bank_length":400},
    "青森":{"bank_angle":32.3,"straight_length":58.9,"bank_length":400},
    "いわき平":{"bank_angle":32.9,"straight_length":62.7,"bank_length":400},
    "弥彦":{"bank_angle":32.4,"straight_length":63.1,"bank_length":400},
    "前橋":{"bank_angle":36.0,"straight_length":46.7,"bank_length":335},
    "取手":{"bank_angle":31.5,"straight_length":54.8,"bank_length":400},
    "宇都宮":{"bank_angle":25.8,"straight_length":63.3,"bank_length":500},
    "大宮":{"bank_angle":26.3,"straight_length":66.7,"bank_length":500},
    "西武園":{"bank_angle":29.4,"straight_length":47.6,"bank_length":400},
    "京王閣":{"bank_angle":32.2,"straight_length":51.5,"bank_length":400},
    "立川":{"bank_angle":31.2,"straight_length":58.0,"bank_length":400},
    "松戸":{"bank_angle":29.8,"straight_length":38.2,"bank_length":333},
    "川崎":{"bank_angle":32.2,"straight_length":58.0,"bank_length":400},
    "平塚":{"bank_angle":31.5,"straight_length":54.2,"bank_length":400},
    "小田原":{"bank_angle":35.6,"straight_length":36.1,"bank_length":333},
    "伊東":{"bank_angle":34.7,"straight_length":46.6,"bank_length":333},
    "静岡":{"bank_angle":30.7,"straight_length":56.4,"bank_length":400},
    "名古屋":{"bank_angle":34.0,"straight_length":58.8,"bank_length":400},
    "岐阜":{"bank_angle":32.3,"straight_length":59.3,"bank_length":400},
    "大垣":{"bank_angle":30.6,"straight_length":56.0,"bank_length":400},
    "豊橋":{"bank_angle":33.8,"straight_length":60.3,"bank_length":400},
    "富山":{"bank_angle":33.7,"straight_length":43.0,"bank_length":333},
    "松坂":{"bank_angle":34.4,"straight_length":61.5,"bank_length":400},
    "四日市":{"bank_angle":32.3,"straight_length":62.4,"bank_length":400},
    "福井":{"bank_angle":31.5,"straight_length":52.8,"bank_length":400},
    "奈良":{"bank_angle":33.4,"straight_length":38.0,"bank_length":333},
    "向日町":{"bank_angle":30.5,"straight_length":47.3,"bank_length":400},
    "和歌山":{"bank_angle":32.3,"straight_length":59.9,"bank_length":400},
    "岸和田":{"bank_angle":30.9,"straight_length":56.7,"bank_length":400},
    "玉野":{"bank_angle":30.6,"straight_length":47.9,"bank_length":400},
    "広島":{"bank_angle":30.8,"straight_length":57.9,"bank_length":400},
    "防府":{"bank_angle":34.7,"straight_length":42.5,"bank_length":333},
    "高松":{"bank_angle":33.3,"straight_length":54.8,"bank_length":400},
    "小松島":{"bank_angle":29.8,"straight_length":55.5,"bank_length":400},
    "高知":{"bank_angle":24.5,"straight_length":52.0,"bank_length":500},
    "松山":{"bank_angle":34.0,"straight_length":58.6,"bank_length":400},
    "小倉":{"bank_angle":34.0,"straight_length":56.9,"bank_length":400},
    "久留米":{"bank_angle":31.5,"straight_length":50.7,"bank_length":400},
    "武雄":{"bank_angle":32.0,"straight_length":64.4,"bank_length":400},
    "佐世保":{"bank_angle":31.5,"straight_length":40.2,"bank_length":400},
    "別府":{"bank_angle":33.7,"straight_length":59.9,"bank_length":400},
    "熊本":{"bank_angle":34.3,"straight_length":60.3,"bank_length":400},
    "手入力":{"bank_angle":30.0,"straight_length":52.0,"bank_length":400},
}
VELODROME_MASTER = {
    "函館":{"lat":41.77694,"lon":140.76283,"home_azimuth":None},
    "青森":{"lat":40.79717,"lon":140.66469,"home_azimuth":None},
    "いわき平":{"lat":37.04533,"lon":140.89150,"home_azimuth":None},
    "弥彦":{"lat":37.70778,"lon":138.82886,"home_azimuth":None},
    "前橋":{"lat":36.39728,"lon":139.05778,"home_azimuth":None},
    "取手":{"lat":35.90175,"lon":140.05631,"home_azimuth":None},
    "宇都宮":{"lat":36.57197,"lon":139.88281,"home_azimuth":None},
    "大宮":{"lat":35.91962,"lon":139.63417,"home_azimuth":None},
    "西武園":{"lat":35.76983,"lon":139.44686,"home_azimuth":None},
    "京王閣":{"lat":35.64294,"lon":139.53372,"home_azimuth":None},
    "立川":{"lat":35.70214,"lon":139.42300,"home_azimuth":None},
    "松戸":{"lat":35.80417,"lon":139.91119,"home_azimuth":None},
    "川崎":{"lat":35.52844,"lon":139.70944,"home_azimuth":None},
    "平塚":{"lat":35.32547,"lon":139.36342,"home_azimuth":None},
    "小田原":{"lat":35.25089,"lon":139.14947,"home_azimuth":None},
    "伊東":{"lat":34.954667,"lon":139.092639,"home_azimuth":None},
    "静岡":{"lat":34.973722,"lon":138.419417,"home_azimuth":None},
    "名古屋":{"lat":35.175560,"lon":136.854028,"home_azimuth":None},
    "岐阜":{"lat":35.414194,"lon":136.783917,"home_azimuth":None},
    "大垣":{"lat":35.361389,"lon":136.628444,"home_azimuth":None},
    "豊橋":{"lat":34.770167,"lon":137.417250,"home_azimuth":None},
    "富山":{"lat":36.757250,"lon":137.234833,"home_azimuth":None},
    "松坂":{"lat":34.564611,"lon":136.533833,"home_azimuth":None},
    "四日市":{"lat":34.965389,"lon":136.634500,"home_azimuth":None},
    "福井":{"lat":36.066889,"lon":136.253722,"home_azimuth":None},
    "奈良":{"lat":34.681111,"lon":135.823083,"home_azimuth":None},
    "向日町":{"lat":34.949222,"lon":135.708389,"home_azimuth":None},
    "和歌山":{"lat":34.228694,"lon":135.171833,"home_azimuth":None},
    "岸和田":{"lat":34.477500,"lon":135.369389,"home_azimuth":None},
    "玉野":{"lat":34.497333,"lon":133.961389,"home_azimuth":None},
    "広島":{"lat":34.359778,"lon":132.502889,"home_azimuth":None},
    "防府":{"lat":34.048778,"lon":131.568611,"home_azimuth":None},
    "高松":{"lat":34.345936,"lon":134.061994,"home_azimuth":None},
    "小松島":{"lat":34.005667,"lon":134.594556,"home_azimuth":None},
    "高知":{"lat":33.566694,"lon":133.526083,"home_azimuth":None},
    "松山":{"lat":33.808889,"lon":132.742333,"home_azimuth":None},
    "小倉":{"lat":33.885722,"lon":130.883167,"home_azimuth":None},
    "久留米":{"lat":33.316667,"lon":130.547778,"home_azimuth":None},
    "武雄":{"lat":33.194083,"lon":130.023083,"home_azimuth":None},
    "佐世保":{"lat":33.161667,"lon":129.712833,"home_azimuth":None},
    "別府":{"lat":33.282806,"lon":131.460472,"home_azimuth":None},
    "熊本":{"lat":32.789167,"lon":130.754722,"home_azimuth":None},
    "手入力":{"lat":None,"lon":None,"home_azimuth":None},
}

# --- 印別実測率（統計） ---
# NOTE: KO（隊列ノックアウト）には使わない。混ぜると「統計が順位をワープさせる」ため。
RANK_STATS_TOTAL = {
    "◎": {"p1": 0.261, "pTop2": 0.459, "pTop3": 0.617},
    "〇": {"p1": 0.235, "pTop2": 0.403, "pTop3": 0.533},
    "▲": {"p1": 0.175, "pTop2": 0.331, "pTop3": 0.484},
    "△": {"p1": 0.133, "pTop2": 0.282, "pTop3": 0.434},
    "×": {"p1": 0.109, "pTop2": 0.242, "pTop3": 0.390},
    "α": {"p1": 0.059, "pTop2": 0.167, "pTop3": 0.295},
    "無": {"p1": 0.003, "pTop2": 0.118, "pTop3": 0.256},
}

def compute_weighted_rank_from_carfr_text(carfr_text: str):
    """
    統計混入スコア（FR×印別実測率）は現在は使用しない。
    互換のため関数だけ残し、常に空を返す。
    """
    return []






# KO(勝ち上がり)関連
KO_GIRLS_SCALE = 0.0
KO_HEADCOUNT_SCALE = {5:0.6, 6:0.8, 7:1.0, 8:1.0, 9:1.0}
KO_GAP_DELTA = 0.007   # 0.010 → 0.007
KO_STEP_SIGMA = 0.35   # 0.4 → 0.35


# ◎ライン格上げ
LINE_BONUS_ON_TENKAI = {"優位"}
LINE_BONUS = {"second": 0.08, "thirdplus": 0.04}
LINE_BONUS_CAP = 0.10
PROB_U = {"second": 0.00, "thirdplus": 0.00}

# --- 安定度（着順分布）をT本体に入れるための重み ---
STAB_W_IN3  = 0.10   # 3着内率の重み
STAB_W_OUT  = 0.12   # 着外率の重み（マイナス補正）
STAB_W_LOWN = 0.05   # サンプル不足補正
STAB_PRIOR_IN3 = 0.33
STAB_PRIOR_OUT = 0.45
def _stab_n0(n: int) -> int:
    """サンプル不足時の事前分布の強さ（nが小さいほど強く効かせる）"""
    if n <= 6: return 12
    if n <= 14: return 8
    if n <= 29: return 5
    return 3
# ==============================
# ユーティリティ
# ==============================
def clamp(x,a,b): return max(a, min(b, x))

def zscore_list(arr):
    arr = np.array(arr, dtype=float)
    m, s = float(np.mean(arr)), float(np.std(arr))
    return np.zeros_like(arr) if s==0 else (arr-m)/s

def zscore_val(x, xs):
    xs = np.array(xs, dtype=float); m, s = float(np.mean(xs)), float(np.std(xs))
    return 0.0 if s==0 else (float(x)-m)/s

# ==============================
# H：最終ホーム地力補正
# ==============================
H_SCORE_SCALE = float(globals().get("H_SCORE_SCALE", 0.045))
H_SCORE_CAP   = float(globals().get("H_SCORE_CAP", 0.075))

def calc_h_score_map(H: dict, active_cars: list[int]) -> dict[int, float]:
    """
    Hをレース内z化して、車番ごとの相対H評価を作る。
    絶対値ではなく、そのレース内でHが高いか低いかを見る。
    """
    vals = np.array(
        [float(H.get(int(n), 0.0)) for n in active_cars],
        dtype=float
    )

    if len(vals) < 2:
        return {int(n): 0.0 for n in active_cars}

    mu = float(np.mean(vals))
    sd = float(np.std(vals))

    if sd < 1e-12:
        return {int(n): 0.0 for n in active_cars}

    return {
        int(n): float((float(H.get(int(n), 0.0)) - mu) / sd)
        for n in active_cars
    }


def h_home_bonus(no: int, role: str, H_Z: dict[int, float]) -> float:
    """
    H補正。
    ライン先頭・単騎を中心に加点。
    番手・三番手は薄く反映。
    """
    role_mult = {
        "head": 1.00,
        "single": 0.70,
        "second": 0.30,
        "thirdplus": 0.15,
    }.get(role, 0.20)

    raw = H_SCORE_SCALE * float(H_Z.get(int(no), 0.0)) * role_mult
    return round(clamp(raw, -H_SCORE_CAP, H_SCORE_CAP), 3)


def h_lead_line_bonus(
    no: int,
    role: str,
    H: dict,
    B: dict,
    line_def: dict,
    home_top_gid,
) -> float:
    """
    H主導ラインの先頭車だけを下支えする補正。
    目的：H主導ライン先頭がKO最下位まで沈む現象を防ぐ。
    """
    try:
        if home_top_gid is None:
            return 0.0

        members = line_def.get(home_top_gid, [])
        if not members:
            return 0.0

        head = int(members[0])

        # H主導ラインの先頭車だけ対象
        if int(no) != head:
            return 0.0

        # 役割が先頭でないなら対象外
        if role != "head":
            return 0.0

        h_val = float(H.get(int(no), 0.0) or 0.0)
        b_val = float(B.get(int(no), 0.0) or 0.0)

        # Hが低いなら補正しない
        if h_val < 3.0:
            return 0.0

        # Hを主、Bを補助にする
        bonus = 0.035 + 0.004 * h_val + 0.002 * b_val

        # 暴走防止
        return round(clamp(bonus, 0.0, 0.090), 3)

    except Exception:
        return 0.0
    raw = H_SCORE_SCALE * float(H_Z.get(int(no), 0.0)) * role_mult
    return round(clamp(raw, -H_SCORE_CAP, H_SCORE_CAP), 3)


def t_score_from_finite(values: np.ndarray, eps: float = 1e-9):
    """NaNを除いた母集団でT=50+10*(x-μ)/σを作り、NaNは50に置換して返す"""
    v = values.astype(float, copy=True)
    finite = np.isfinite(v)
    k = int(finite.sum())
    if k < 2:
        return np.full_like(v, 50.0), (float("nan") if k==0 else float(v[finite][0])), 0.0, k
    mu = float(np.mean(v[finite]))
    sd = float(np.std(v[finite], ddof=0))
    if (not np.isfinite(sd)) or sd < eps:
        return np.full_like(v, 50.0), mu, 0.0, k
    T = 50.0 + 10.0 * ((v - mu) / sd)
    T[~finite] = 50.0
    return T, mu, sd, k

def extract_car_list(s, n_cars=None):
    """
    ライン入力文字列から車番を抽出する。

    v179修正：
    ・単騎の「6」も [6] として必ず有効扱いする。
    ・2桁以上のラインだけを有効にする判定は行わない。
    ・出走数 n_cars では車番を制限しない。
      5車立てでも欠番あり入力を許可するため。
    ・同一ライン内の重複は先頭1回だけ残す。
    """
    cars = []
    seen = set()

    raw = "" if s is None else str(s)
    for ch in raw:
        if not ch.isdigit():
            continue
        v = int(ch)
        if 1 <= v <= 9 and v not in seen:
            cars.append(v)
            seen.add(v)

    return cars


def build_line_maps(lines):
    # 最大9ラインまで対応。単騎も1ライン。
    labels = "ABCDEFGHI"
    line_def = {}
    for i, lst in enumerate(lines):
        if not lst:
            continue
        label = labels[i] if i < len(labels) else f"L{i+1}"
        line_def[label] = list(lst)
    car_to_group = {c: g for g, mem in line_def.items() for c in mem}
    return line_def, car_to_group


def _format_lines_for_check(lines):
    """入力確認用：[[7,1,4],[5,3,2],[6]] → '714 / 532 / 6'"""
    try:
        parts = []
        for lst in lines:
            if not lst:
                continue
            parts.append("".join(str(int(x)) for x in lst))
        return " / ".join(parts) if parts else "—"
    except Exception:
        return "—"


def role_in_line(car, line_def):
    for g, mem in line_def.items():
        if car in mem:
            if len(mem) == 1:
                return 'single'
            idx = mem.index(car)
            return ['head', 'second', 'thirdplus'][idx] if idx < 3 else 'thirdplus'
    return 'single'
# =====================================================
# ラスト半周補正：番手差し・前で動ける上位補正
# =====================================================

LAST_HALF_ENABLE = True

# ラスト半周補正の全体上限
LAST_HALF_CAP = 0.050

# 番手補正の上限
LAST_HALF_SECOND_CAP = 0.050

# 先頭・単騎の前で動ける補正の上限
LAST_HALF_FRONT_CAP = 0.040


def _is_top_third(rank_val, top_third_limit: int) -> bool:
    """
    レース内上位1/3判定。
    7車なら3位以内。
    """
    try:
        return int(rank_val) <= int(top_third_limit)
    except Exception:
        return False


def calc_last_half_role_bonus(
    role: str,
    kaku: str,
    tenscore: float,
    leader_tenscore: float,
    race_avg_tenscore: float,
    h_count: float = 0.0,
    b_count: float = 0.0,
    race_score_rank=None,
    ko_score_rank=None,
    tenkai_score_rank=None,
    top_third_limit: int = 3,
    scenario_top_count: int = 0,
    p1_rate=None,
    p2_rate=None,
    p3_rate=None,
):
    """
    ラスト半周〜ゴール前の個人戦補正。

    思想：
    ラスト半周までは団体戦。
    ラスト半周からは個人戦。
    そのため、位置ではなく「実際に着を取れる個人成績」で補正する。

    使用するもの：
    ・1着率
    ・2着内率
    ・3着内率

    使わないもの：
    ・番手位置だけの加点
    ・H/Bだけの加点
    ・自力だから加点
    ・単騎だから加点
    ・H主導3番手以降だから加点
    """

    if not LAST_HALF_ENABLE:
        return 0.0, []

    bonus = 0.0
    reasons = []

    try:
        role = str(role)

        def _rate(v):
            try:
                x = float(v)
                if x > 1.0:
                    x = x / 100.0
                return x
            except Exception:
                return None

        p1 = _rate(p1_rate)
        p2 = _rate(p2_rate)
        p3 = _rate(p3_rate)

        # ---------------------------------------------
        # 個人戦補正
        # ---------------------------------------------
        # 勝ち切れる個人力を強めに評価
        if p1 is not None and p1 >= 0.20:
            bonus += 0.025
            reasons.append(f"1着率{p1 * 100:.0f}%以上")

        # 2着内率は評価するが、1着率より軽くする
        if p2 is not None and p2 >= 0.30:
            bonus += 0.010
            reasons.append(f"2着内率{p2 * 100:.0f}%以上")

        # 3着内率は、2着内率もある場合だけ補正
        # 3着に残るだけの選手をラスト半周個人力として過大評価しない
        if (
            p3 is not None
            and p3 >= 0.40
            and p2 is not None
            and p2 >= 0.30
        ):
            bonus += 0.010
            reasons.append(f"3着内率{p3 * 100:.0f}%以上")

        # ---------------------------------------------
        # 役割別上限
        # 位置で加点はしない。
        # ただし3番手以降だけは暴走防止で上限を低くする。
        # ---------------------------------------------
        if role == "thirdplus":
            role_cap = 0.030
        else:
            role_cap = 0.050

        bonus = clamp(bonus, 0.0, role_cap)
        bonus = clamp(bonus, -LAST_HALF_CAP, LAST_HALF_CAP)

        if not reasons:
            reasons.append("補正なし")

        return round(float(bonus), 3), reasons

    except Exception as e:
        return 0.0, [f"ラスト半周補正エラー:{e}"]

# ==============================

# =====================================================
# 混戦度判定
#   平均得点ではなく、競走得点1位と2位の差で見る
#   High   = 上位差が大きく、順当寄り
#   Middle = 標準
#   Low    = 上位差が小さく、波乱気味
#
#   ※スコア補正には使わない。表示・検証用。
# =====================================================
def calc_race_compactness(ratings_val: dict, active_cars: list):
    vals = []

    for no in active_cars:
        try:
            v = float(ratings_val.get(int(no), 0.0))
            if v > 0:
                vals.append(v)
        except Exception:
            pass

    if len(vals) < 2:
        return {
            "label": "未判定",
            "top1": 0.0,
            "top2": 0.0,
            "top_gap": None,
        }

    vals = sorted(vals, reverse=True)

    top1 = vals[0]
    top2 = vals[1]
    top_gap = top1 - top2

    if top_gap >= 2.00:
        label = "順当寄り"
    elif top_gap >= 1.00:
        label = "標準"
    else:
        label = "波乱気味"

    return {
        "label": label,
        "top1": float(top1),
        "top2": float(top2),
        "top_gap": float(top_gap),
    }

# H：最終ホーム想定ライン
# ==============================
def calc_home_line_scores(line_def: dict, H: dict, B: dict, active_cars: list[int]) -> dict:
    """
    H = 最終ホーム先頭通過回数を使って、
    最終周回ホームで前に出やすいラインを評価する。
    ※本体スコアには混ぜず、展開表示用。
    """
    scores = {}

    for gid, members in line_def.items():
        mem = [int(x) for x in members if int(x) in active_cars]
        if not mem:
            continue

        head = mem[0]
        second = mem[1] if len(mem) >= 2 else None
        third = mem[2] if len(mem) >= 3 else None

        head_h = float(H.get(head, 0))
        second_h = float(H.get(second, 0)) if second is not None else 0.0
        third_h = float(H.get(third, 0)) if third is not None else 0.0

        # 単騎は自分のHをそのまま見る
        if len(mem) == 1:
            score = head_h
        else:
            # ライン先頭のHを主役、番手・三番手は補助
            score = head_h * 0.75 + second_h * 0.15 + third_h * 0.10

        # 同点時の微差用：Bをほんの少しだけ見る
        score += float(B.get(head, 0)) * 0.01

        scores[gid] = round(score, 3)

    return scores


def make_home_line_order(line_def: dict, H: dict, B: dict, active_cars: list[int]) -> list:
    """
    最終ホーム想定ライン順を返す。
    """
    scores = calc_home_line_scores(line_def, H, B, active_cars)

    return sorted(
        scores.keys(),
        key=lambda gid: scores.get(gid, 0.0),
        reverse=True
    )


def format_home_line_order(line_def: dict, order: list) -> str:
    """
    A/B/Cなどのgid順を、実際のライン文字列に変換する。
    例：['B','C','A'] → 26　37　145
    """
    parts = []

    for gid in order:
        members = line_def.get(gid, [])
        if members:
            parts.append("".join(str(int(x)) for x in members))

    return "　".join(parts) if parts else "—"


# 単騎を全体的に抑える共通係数（あとでいじれるようにする）
SINGLE_NERF = float(globals().get("SINGLE_NERF", 0.85))  # 0.80〜0.88くらいで調整

def pos_coeff(role, line_factor):
    base_map = {
        'head':      1.00,
        'second':    0.72,   # 0.70→0.72に少し上げてライン2番手をちゃんと評価
        'thirdplus': 0.55,
        'single':    0.52,   # 0.90 → 0.52 にドンと落とす
    }
    base = base_map.get(role, 0.52)
    if role == 'single':
        base *= SINGLE_NERF      # ここでさらに細かく落とせる
    return base * line_factor


def tenscore_correction(tenscores):
    n = len(tenscores)
    if n<=2: return [0.0]*n
    df = pd.DataFrame({"得点":tenscores})
    df["順位"] = df["得点"].rank(ascending=False, method="min").astype(int)
    hi = min(n,8)
    baseline = df[df["順位"].between(2,hi)]["得点"].mean()
    def corr(row):
        return round(abs(baseline-row["得点"])*0.03, 3) if row["順位"] in [2,3,4] else 0.0
    return df.apply(corr, axis=1).tolist()

def track_effective_ratio(track_name: str,
                           alpha_goal: float = 0.50,
                           beta_corner: float = 0.25) -> float:
    d = KEIRIN_DATA.get(track_name)
    if not d:
        return 0.50
    lap  = float(d.get("bank_length", 400))
    home = float(d.get("straight_length", 52.0))
    back = 2.0 * home  # ゴール前は半分の仮定
    corner_total = max(lap - home - back, 0.0)
    L_eff = back + alpha_goal * home + beta_corner * corner_total
    ratio = (L_eff / lap) if lap > 0 else 0.50
    return clamp(ratio, 0.20, 0.90)


# =====================================================
# 会場成績手入力補正 × 最終ホームライン流れ補正
#   入力例：
#     的中率 = 12/40     → 30.0%
#     回収率 = 12000/8000 → 150.0%
#   思想：
#     成績が悪い会場ほど、最終H1番手ライン先頭のイン減速を疑い、
#     最終H2番手ライン、とくに番手の外スピード差しを評価する。
# =====================================================

def parse_fraction_rate(text: str, percent: bool = True):
    """
    '12/40' や '12000/8000' を率に変換する。
    percent=True なら 30.0 のように％値で返す。
    空欄・不正値・分母0は None。
    """
    s = str(text or "").strip()
    if not s:
        return None

    try:
        if "/" in s:
            a, b = s.split("/", 1)
            a = float(str(a).replace(",", "").strip())
            b = float(str(b).replace(",", "").strip())
            if b <= 0:
                return None
            rate = a / b
        else:
            v = float(s.replace("%", "").replace(",", "").strip())
            rate = v / 100.0 if v > 1.0 else v

        if not math.isfinite(rate):
            return None

        return rate * 100.0 if percent else rate

    except Exception:
        return None


def judge_venue_profile(hit_rate, return_rate):
    """
    hit_rate / return_rate は％値。
    例：30.0, 120.0
    """
    hr = None if hit_rate is None else float(hit_rate)
    rr = None if return_rate is None else float(return_rate)

    if hr is None and rr is None:
        return "unknown"

    # 回収率が強い。的中率が低ければ一撃型。
    if rr is not None and rr >= 100.0:
        if hr is None or hr >= 35.0:
            return "strong_good"
        return "swing_return"

    # 的中しているのに安い。順位は壊さず必要オッズ側で締める。
    if hr is not None and hr >= 31.0 and rr is not None and rr < 80.0:
        return "cheap_hit"

    # 的中率がかなり低い。
    if hr is not None and hr < 22.0:
        if rr is not None and rr < 50.0:
            return "very_bad"
        return "low_hit_risk"

    # 回収率がかなり悪い。
    if rr is not None and rr < 50.0:
        return "bad"

    # 回収率が低め。
    if rr is not None and rr < 70.0:
        return "normal_watch"

    return "normal"


def _venue_fit_hit_coef(hit_rate):
    """
    v203:
    会場別の的中率を、2車複の「的中期待」へ小幅倍率として反映する。
    hit_rate は％値（例：25.3）。未入力時は 1.00。

    強くしすぎると会場判定に振り回されるため、概ね 0.90〜1.08 に収める。
    """
    try:
        if hit_rate is None:
            return 1.00
        hr = float(hit_rate)
        if not math.isfinite(hr):
            return 1.00
        if hr >= 35.0:
            return 1.08
        if hr >= 30.0:
            return 1.04
        if hr >= 25.0:
            return 1.00
        if hr >= 22.0:
            return 0.96
        if hr >= 18.0:
            return 0.92
        return 0.90
    except Exception:
        return 1.00


def _venue_fit_myoumi_coef(return_rate):
    """
    v203:
    会場別の回収率を、2車複の「妙味期待」へ小幅倍率として反映する。
    return_rate は％値（例：75.5）。未入力時は 1.00。

    回収率が低い開催では、妙味A++頼みの買目が自然に下がる。
    逆に回収率が高い開催では、妙味期待を少し信頼する。
    """
    try:
        if return_rate is None:
            return 1.00
        rr = float(return_rate)
        if not math.isfinite(rr):
            return 1.00
        if rr >= 120.0:
            return 1.10
        if rr >= 100.0:
            return 1.06
        if rr >= 85.0:
            return 1.00
        if rr >= 70.0:
            return 0.94
        if rr >= 50.0:
            return 0.90
        return 0.88
    except Exception:
        return 1.00


def _fmt_venue_fit_coef(v):
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "1.00"


VENUE_HOME_FLOW_MULT = {
    "strong_good": 0.50,
    "swing_return": 0.85,
    "normal": 1.00,
    "normal_watch": 1.10,
    "cheap_hit": 0.90,
    "bad": 1.25,
    "low_hit_risk": 1.35,
    "very_bad": 1.50,
    "unknown": 1.00,
}

VENUE_MIN_ODDS_MULT = {
    "strong_good": 0.95,
    "swing_return": 1.05,
    "normal": 1.00,
    "normal_watch": 1.10,
    "cheap_hit": 1.25,
    "bad": 1.20,
    "low_hit_risk": 1.30,
    "very_bad": 1.40,
    "unknown": 1.00,
}

# 係数は「補正点」ではなく倍率前の思想値。
# 実際は HOME_FLOW_BASE_SCALE と会場倍率を掛けて使う。
HOME_FLOW_BASE_SCALE = 0.04
HOME_FLOW_COEF = {
    "top_line": {
        "head":      -0.70,
        "second":    -0.20,
        "third":     -0.10,
        "single":    -0.50,
    },
    "second_line": {
        "head":      +0.50,
        "second":    +0.70,
        "third":     +0.25,
        "single":    +0.20,
    },
    "other_line": {
        "head":       0.00,
        "second":     0.00,
        "third":      0.00,
        "single":     0.00,
    },
}


def calc_venue_shape_index(track_name: str):
    """
    バンク形状から、長いみなし直線リスクを軽く算出する。
    会場成績の補助係数として使い、実績入力を主にする。
    """
    d = KEIRIN_DATA.get(track_name)
    if not d:
        return {"minashi_ratio": 0.0, "bank_support": 0.0, "stretch_risk": 0.0}

    angle = float(d.get("bank_angle", 30.0) or 30.0)
    straight = float(d.get("straight_length", 52.0) or 52.0)
    bank = float(d.get("bank_length", 400.0) or 400.0)

    minashi = 1.75 * straight + 0.25 * bank
    minashi_ratio = minashi / max(bank, 1e-9)
    bank_support = angle / max(minashi_ratio, 1e-9)

    stretch_risk = 0.0
    if minashi_ratio >= 0.520:
        stretch_risk += 1.00
    elif minashi_ratio >= 0.510:
        stretch_risk += 0.60
    elif minashi_ratio >= 0.500:
        stretch_risk += 0.30

    if bank_support < 62.5:
        stretch_risk += 0.60
    elif bank_support < 65.0:
        stretch_risk += 0.30

    if bank <= 340:
        stretch_risk *= 0.75

    return {
        "minashi_ratio": round(float(minashi_ratio), 6),
        "bank_support": round(float(bank_support), 3),
        "stretch_risk": round(float(clamp(stretch_risk, 0.0, 1.50)), 3),
    }


def venue_home_flow_multiplier(track_name: str, venue_profile: str) -> float:
    """
    会場成績による倍率を主、バンク形状リスクを従として合成する。
    strong_good は元評価を壊さないため弱く、very_bad は強くする。
    """
    profile_mult = float(VENUE_HOME_FLOW_MULT.get(str(venue_profile), 1.00))

    try:
        shape = calc_venue_shape_index(track_name)
        shape_risk = float(shape.get("stretch_risk", 0.0) or 0.0)
    except Exception:
        shape_risk = 0.0

    shape_mult = 1.00 + 0.10 * shape_risk
    return round(clamp(profile_mult * shape_mult, 0.40, 1.80), 3)


def home_flow_adjust_by_venue(
    no: int,
    role: str,
    gid,
    home_top_gid,
    home_second_gid,
    track_name: str,
    venue_profile: str,
):
    """
    最終ホーム想定ライン補正。
    - 1番手ライン：イン減速リスクとして減点。特に先頭。
    - 2番手ライン：外スピードラインとして加点。特に番手。
    - その他：据え置き。
    """
    if gid is None:
        return 0.0, "ライン不明"

    if gid == home_top_gid:
        line_pos = "top_line"
        line_label = "H1番手"
    elif home_second_gid is not None and gid == home_second_gid:
        line_pos = "second_line"
        line_label = "H2番手"
    else:
        line_pos = "other_line"
        line_label = "その他"

    r = str(role or "single")
    if r == "thirdplus":
        r = "third"

    mult = venue_home_flow_multiplier(track_name, venue_profile)
    scale = float(HOME_FLOW_BASE_SCALE) * float(mult)
    coef = float(HOME_FLOW_COEF.get(line_pos, {}).get(r, 0.0))
    adj = round(coef * scale, 3)

    reason = f"{line_label}/{r} 係数{coef:+.2f}×倍率{mult:.2f}"
    return adj, reason



def wind_adjust(wind_dir, wind_speed, role, prof_escape):
    s = max(0.0, float(wind_speed))
    WIND_ZERO   = float(globals().get("WIND_ZERO", 0.0))
    WIND_SIGN   = float(globals().get("WIND_SIGN", 1.0))
    WIND_GAIN   = float(globals().get("WIND_GAIN", 1.0))
    WIND_CAP    = float(globals().get("WIND_CAP", 0.06))
    WIND_MODE   = globals().get("WIND_MODE", "scalar")
    WIND_COEFF  = globals().get("WIND_COEFF", {})
    SPECIAL_DIRECTIONAL_VELODROMES = globals().get("SPECIAL_DIRECTIONAL_VELODROMES", set())

    try:
        s_state_track = st.session_state.get("track", "")
    except Exception:
        s_state_track = ""

    # --- 風速→基礎量 ---
    if s <= WIND_ZERO:
        base = 0.0
    elif s <= 5.0:
        base = 0.006 * (s - WIND_ZERO)
    elif s <= 8.0:
        base = 0.021 + 0.008 * (s - 5.0)
    else:
        base = 0.045 + 0.010 * min(s - 8.0, 4.0)

    # --- 位置係数 ---
    pos = {'head':1.00,'second':0.85,'single':0.75,'thirdplus':0.65}.get(role, 0.75)

    # ===== ★ここ①：強風ほど番手・後位を不利にする =====
    wind01 = clamp((s - WIND_ZERO) / (8.0 - WIND_ZERO), 0.0, 1.0)
    track_ratio = track_effective_ratio(s_state_track)
    wind_eff01 = wind01 * track_ratio

    if role in ("second", "thirdplus"):
        pos *= (1.0 - 0.20 * wind_eff01)   # 最大20%だけ削る

    # --- 脚質（自力） ---
    prof = 0.35 + 0.65 * float(prof_escape)
    val = base * pos * prof

    # --- 風向き（既存） ---
    if (WIND_MODE == "directional") or (s >= 7.0 and s_state_track in SPECIAL_DIRECTIONAL_VELODROMES):
        wd = WIND_COEFF.get(wind_dir, 0.0)
        dir_term = clamp(
            s * wd * (0.30 + 0.70 * float(prof_escape)) * 0.6,
            -0.03, 0.03
        )
        val += dir_term

    # ===== ★ここ②：会場ごとに風の効きをスケール =====
    val *= clamp(track_ratio / 0.50, 0.60, 1.40)

    val = (val * float(WIND_SIGN)) * float(WIND_GAIN)
    return round(clamp(val, -float(WIND_CAP), float(WIND_CAP)), 3)


# === 直線ラスト200m（残脚）補正｜33バンク対応版 ==============================
# 33（<=340m）は「先行ペナ弱め／差し・追込ボーナス控えめ」へ最適化
L200_ESC_PENALTY = float(globals().get("L200_ESC_PENALTY", -0.06))  # 先行は垂れやすい（基本）
L200_SASHI_BONUS = float(globals().get("L200_SASHI_BONUS", +0.03))  # 差しは伸びやすい
L200_MARK_BONUS  = float(globals().get("L200_MARK_BONUS",  +0.02))  # 追込は少し上げ

L200_GRADE_GAIN  = globals().get("L200_GRADE_GAIN", {
    "F2": 1.18, "F1": 1.10, "G": 1.05, "GIRLS": 0.95, "TOTAL": 1.00
})

# 短走路増幅：旧1.15 → 33はむしろ緩和（0.85）
L200_SHORT_GAIN_33   = float(globals().get("L200_SHORT_GAIN_33", 0.85))
L200_SHORT_GAIN_OTH  = float(globals().get("L200_SHORT_GAIN_OTH", 1.00))
L200_LONG_RELAX      = float(globals().get("L200_LONG_RELAX", 0.90))
L200_CAP             = float(globals().get("L200_CAP", 0.08))
L200_WET_GAIN        = float(globals().get("L200_WET_GAIN", 1.15))

# 33専用 成分別スケーリング
L200_33_ESC_MULT   = float(globals().get("L200_33_ESC_MULT", 0.80))  # 逃ペナ 20%縮小
L200_33_SASHI_MULT = float(globals().get("L200_33_SASHI_MULT", 0.85))# 差し  15%縮小
L200_33_MARK_MULT  = float(globals().get("L200_33_MARK_MULT", 0.90)) # 追込  10%縮小

def _grade_key_from_class(race_class: str) -> str:
    if "ガール" in race_class: return "GIRLS"
    if "Ｓ級" in race_class or "S級" in race_class: return "G"
    if "チャレンジ" in race_class: return "F2"
    if "Ａ級" in race_class or "A級" in race_class: return "F1"
    return "TOTAL"

def l200_adjust(role: str,
                straight_length: float,
                bank_length: float,
                race_class: str,
                prof_escape: float,    # 逃
                prof_sashi: float,     # 差
                prof_oikomi: float,    # マ
                is_wet: bool = False) -> float:
    """
    ラスト200mの“残脚”を脚質×バンク×グレードで調整した無次元値（±）を返す。
    ※ ENV合計（total_raw）には足さず、独立柱として z 化→anchor_score へ。
    """
    esc_term   = L200_ESC_PENALTY * float(prof_escape)
    sashi_term = L200_SASHI_BONUS * float(prof_sashi)
    mark_term  = L200_MARK_BONUS  * float(prof_oikomi)

    is_33 = float(bank_length) <= 340.0
    if is_33:
        esc_term   *= L200_33_ESC_MULT
        sashi_term *= L200_33_SASHI_MULT
        mark_term  *= L200_33_MARK_MULT

    base = esc_term + sashi_term + mark_term

    if is_33:
        base *= L200_SHORT_GAIN_33
    else:
        base *= L200_SHORT_GAIN_OTH

    if float(straight_length) >= 60.0:
        base *= L200_LONG_RELAX

    base *= float(L200_GRADE_GAIN.get(_grade_key_from_class(race_class), 1.0))

    if is_wet:
        base *= L200_WET_GAIN

    pos_factor = {'head':1.00,'second':0.85,'thirdplus':0.70,'single':0.80}.get(role, 0.80)
    base *= pos_factor

    return round(clamp(base, -float(L200_CAP), float(L200_CAP)), 3)


def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi, bank_length=None):
    straight_factor = (float(straight_length)-40.0)/10.0
    angle_factor = (float(bank_angle)-25.0)/5.0
    total = clamp(-0.1*straight_factor + 0.1*angle_factor, -0.05, 0.05)
    return round(total*prof_escape - 0.5*total*prof_sashi, 3)

def bank_length_adjust(bank_length, prof_oikomi):
    delta = clamp((float(bank_length)-411.0)/100.0, -0.05, 0.05)
    return round(delta*prof_oikomi, 3)

# --- ラインSBボーナス（33mは自動で半減） --------------------
def compute_lineSB_bonus(line_def, S, B, line_factor=1.0, exclude=None, cap=0.06, enable=True):
    """
    33m系（<=340）では自動で効きを半減:
      - LINE_SB_33_MULT（既定0.5）を line_factor に乗算
      - LINE_SB_CAP_33_MULT（既定0.5）を cap に乗算
    """
    if not enable or not line_def:
        return ({g: 0.0 for g in line_def.keys()} if line_def else {}), {}

    # 33かどうかの自動推定
    try:
        bank_len = st.session_state.get("bank_length", st.session_state.get("track_length", None))
    except Exception:
        bank_len = globals().get("BANK_LENGTH", None)

    eff_line_factor = float(line_factor)
    eff_cap = float(cap)

    if bank_len is not None:
        try:
            if float(bank_len) <= 340.0:
                mult = float(globals().get("LINE_SB_33_MULT", 0.50))
                capm = float(globals().get("LINE_SB_CAP_33_MULT", 0.50))
                eff_line_factor *= mult
                eff_cap *= capm
        except Exception:
            pass

    # ライン内の位置重み（単騎を下げる）
    w_pos_base = {
        "head":      1.00,
        "second":    0.55,
        "thirdplus": 0.38,
        "single":    0.34,
    }

    # ラインごとのS/B集計
    Sg = {}
    Bg = {}
    for g, mem in line_def.items():
        s = 0.0
        b = 0.0
        for car in mem:
            if exclude is not None and car == exclude:
                continue
            role = role_in_line(car, line_def)
            w = w_pos_base[role] * eff_line_factor
            s += w * float(S.get(car, 0))
            b += w * float(B.get(car, 0))
        Sg[g] = s
        Bg[g] = b

    # ラインごとの“強さ”スコア
    raw = {}
    for g in line_def.keys():
        s = Sg[g]
        b = Bg[g]
        ratioS = s / (s + b + 1e-6)
        raw[g] = (0.6 * b + 0.4 * s) * (0.6 + 0.4 * ratioS)

    # z化してボーナス化
    zz = zscore_list(list(raw.values())) if raw else []
    bonus = {}
    for i, g in enumerate(raw.keys()):
        bonus[g] = clamp(0.02 * float(zz[i]), -eff_cap, eff_cap)

    return bonus, raw


# ==============================
# KO Utilities（ここから下を1かたまりで）
# ==============================

def _role_of(car, mem):
    """ラインの中での役割を返す（head / second / thirdplus / single）"""
    if len(mem) == 1:
        return "single"
    idx = mem.index(car)
    return ["head", "second", "thirdplus"][idx] if idx < 3 else "thirdplus"


# KOでも、ライン強度でも、同じ位置重みを使う
LINE_W_POS = {
    "head":      1.00,
    "second":    0.55,
    "thirdplus": 0.38,
    "single":    0.34,
}


def _line_strength_raw(line_def, S, B, line_factor: float = 1.0) -> dict:
    """
    KOやトップ2ライン抽出で使う“生のライン強度”
    compute_lineSB_bonus と式をそろえてある
    """
    if not line_def:
        return {}

    w_pos = {k: v * float(line_factor) for k, v in LINE_W_POS.items()}

    raw: dict[str, float] = {}
    for g, mem in line_def.items():
        s = 0.0
        b = 0.0
        for c in mem:
            role = _role_of(c, mem)
            w = w_pos.get(role, 0.34)
            s += w * float(S.get(c, 0))
            b += w * float(B.get(c, 0))
        ratioS = s / (s + b + 1e-6)
        raw[g] = (0.6 * b + 0.4 * s) * (0.6 + 0.4 * ratioS)
    return raw


def _top2_lines(line_def, S, B, line_factor=1.0):
    """ラインの中から強い2本を取る"""
    raw = _line_strength_raw(line_def, S, B, line_factor)
    order = sorted(raw.keys(), key=lambda g: raw[g], reverse=True)
    return (order[0], order[1]) if len(order) >= 2 else (order[0], None) if order else (None, None)


def _extract_role_car(line_def, gid, role_name):
    """指定ラインのhead/secondを抜く"""
    if gid is None or gid not in line_def:
        return None
    mem = line_def[gid]
    if role_name == "head":
        return mem[0] if len(mem) >= 1 else None
    if role_name == "second":
        return mem[1] if len(mem) >= 2 else None
    return None


def _ko_order(v_base_map,
              line_def,
              S,
              B,
              line_factor: float = 1.0,
              gap_delta: float = 0.007):
    """
    KO用の並びを作る
    1) 上2ラインのhead
    2) 上2ラインのsecond
    3) 残りのラインの残りをスコア順
    4) その他の車番
    同じライン内でスコア差が gap_delta 以内なら寄せる
    """
    cars = list(v_base_map.keys())

    # ラインが無いときはふつうにスコア順
    if not line_def or len(line_def) < 1:
        return [c for c, _ in sorted(v_base_map.items(), key=lambda x: x[1], reverse=True)]

    g1, g2 = _top2_lines(line_def, S, B, line_factor)

    head1 = _extract_role_car(line_def, g1, "head")
    head2 = _extract_role_car(line_def, g2, "head")
    sec1  = _extract_role_car(line_def, g1, "second")
    sec2  = _extract_role_car(line_def, g2, "second")

    others: list[int] = []
    if g1:
        mem = line_def[g1]
        if len(mem) >= 3:
            others += mem[2:]
    if g2:
        mem = line_def[g2]
        if len(mem) >= 3:
            others += mem[2:]
    for g, mem in line_def.items():
        if g not in {g1, g2}:
            others += mem

    order: list[int] = []

    # 1) headをスコア順で
    head_pair = [x for x in [head1, head2] if x is not None]
    order += sorted(head_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)

    # 2) secondをスコア順で
    sec_pair = [x for x in [sec1, sec2] if x is not None]
    order += sorted(sec_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)

    # 3) 残りラインの残り（重複を落とす）
    others = list(dict.fromkeys([c for c in others if c is not None]))
    others_sorted = sorted(others, key=lambda c: v_base_map.get(c, -1e9), reverse=True)
    order += [c for c in others_sorted if c not in order]

    # 4) まだ出てない車を最後に
    for c in cars:
        if c not in order:
            order.append(c)

    # ライン内の小差詰め
    def _same_group(a, b):
        if a is None or b is None:
            return False
        ga = next((g for g, mem in line_def.items() if a in mem), None)
        gb = next((g for g, mem in line_def.items() if b in mem), None)
        return ga is not None and ga == gb

        i = 0
    while i < len(order) - 2:
        a, b, c = order[i], order[i + 1], order[i + 2]
        if _same_group(a, b):
            vx = v_base_map.get(b, 0.0) - v_base_map.get(c, 0.0)
            # b と c の差が小さいなら入れ替えて “寄せる”
            if vx >= -gap_delta:
                order[i + 1], order[i + 2] = order[i + 2], order[i + 1]
        i += 1

    return order


def _zone_from_p(p: float):
    needed = 1.0 / max(p, 1e-12)
    return needed, needed * (1.0 + E_MIN), needed * (1.0 + E_MAX)


def apply_anchor_line_bonus(score_raw: dict[int, float],
                            line_of: dict[int, str],   # ★ int→str に直す
                            role_map: dict[int, str],
                            anchor: int,
                            tenkai: str) -> dict[int, float]:


    a_line = line_of.get(anchor, None)
    is_on = (tenkai in LINE_BONUS_ON_TENKAI) and (a_line is not None)
    score_adj: dict[int, float] = {}
    for i, s in score_raw.items():
        bonus = 0.0
        if is_on and line_of.get(i) == a_line and i != anchor:
            role = role_map.get(i, "single")
            bonus = min(max(0.0, LINE_BONUS.get(role, 0.0)), LINE_BONUS_CAP)
        score_adj[i] = s + bonus
    return score_adj


from typing import Optional, Dict

def format_rank_all(score_map: Dict[int, float], P_floor_val: Optional[float] = None) -> str:
    order = sorted(score_map.keys(), key=lambda k: (-score_map[k], k))
    rows = []
    for i in order:
        if P_floor_val is None:
            rows.append(f"{i}")
        else:
            rows.append(f"{i}" if score_map[i] >= P_floor_val else f"{i}(P未満)")
    return " ".join(rows)




# ==============================
# 風の自動取得（Open-Meteo / 時刻固定）
# 風向は手入力運用のため、APIでは風速だけ取得する軽量版
# ==============================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openmeteo_hour(lat, lon, target_dt_naive):
    """
    Open-Meteoから風速だけ取得する軽量版。
    風向きはVeloBi側で手入力する前提なので取得しない。
    同じ場・同じ日時は1時間キャッシュして429を避ける。
    """
    import numpy as np

    d = target_dt_naive.strftime("%Y-%m-%d")
    base = "https://api.open-meteo.com/v1/forecast"

    url = (
        f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
        "&hourly=wind_speed_10m,precipitation,weather_code"
        "&timezone=Asia%2FTokyo"
        "&windspeed_unit=ms"
        f"&start_date={d}&end_date={d}"
    )

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()

        j = r.json().get("hourly", {})
        times = [datetime.fromisoformat(t) for t in j.get("time", [])]

        if not times:
            raise RuntimeError("empty hourly times")

        diffs = [abs((t - target_dt_naive).total_seconds()) for t in times]
        k = int(np.argmin(diffs))

        sp = j.get("wind_speed_10m", [])
        pr = j.get("precipitation", [])
        wc = j.get("weather_code", [])

        speed = float(sp[k]) if k < len(sp) and sp[k] is not None else float("nan")
        precip = float(pr[k]) if k < len(pr) and pr[k] is not None else 0.0
        weather_code = int(wc[k]) if k < len(wc) and wc[k] is not None else None

        return {
            "time": times[k],
            "speed_ms": speed,
            "deg": None,
            "precipitation": precip,
            "weather_code": weather_code,
            "diff_min": diffs[k] / 60.0,
        }

    except requests.exceptions.HTTPError as e:
        if getattr(e.response, "status_code", None) == 429:
            raise RuntimeError(
                "Open-Meteoの取得制限中です。少し時間を空けるか、手入力の風速を使ってください。"
            )
        raise RuntimeError(f"Open-Meteo取得失敗：{e}")

    except Exception as e:
        raise RuntimeError(f"Open-Meteo取得失敗：{e}")

# ==============================
# サイドバー：開催情報 / バンク・風・頭数
# ==============================

# --- 会場差分（得意会場平均を標準）ヘルパー（このブロック内に自己完結）
FAVORABLE_VENUES = ["名古屋","いわき平","前橋","立川","宇都宮","岸和田","高知"]

def _std_from_venues(names):
    Ls = [KEIRIN_DATA[v]["straight_length"] for v in names if v in KEIRIN_DATA]
    Th = [KEIRIN_DATA[v]["bank_angle"]      for v in names if v in KEIRIN_DATA]
    Cs = [KEIRIN_DATA[v]["bank_length"]     for v in names if v in KEIRIN_DATA]
    return (float(np.mean(Th)), float(np.mean(Ls)), float(np.mean(Cs)))

TH_STD, L_STD, C_STD = _std_from_venues(FAVORABLE_VENUES)

_ALL_L  = np.array([KEIRIN_DATA[k]["straight_length"] for k in KEIRIN_DATA], float)
_ALL_TH = np.array([KEIRIN_DATA[k]["bank_angle"]      for k in KEIRIN_DATA], float)
SIG_L  = float(np.std(_ALL_L))  if np.std(_ALL_L)  > 1e-9 else 1.0
SIG_TH = float(np.std(_ALL_TH)) if np.std(_ALL_TH) > 1e-9 else 1.0

def venue_z_terms(straight_length: float, bank_angle: float, bank_length: float):
    zL  = (float(straight_length) - L_STD)  / SIG_L
    zTH = (float(bank_angle)      - TH_STD) / SIG_TH
    if bank_length >= 480: dC = +0.4
    elif bank_length >= 380: dC = 0.0
    else: dC = -0.4
    return zL, zTH, dC

def venue_mix(zL, zTH, dC):
    # 直線長↑＝差し/捲り寄り(−)、カント↑＝先行/スピード勝負(+)、333短周長＝ライン寄り(−)
    return float(clamp(0.50*zTH - 0.35*zL - 0.30*dC, -1.0, +1.0))


# ==============================
# ★ 風取得ユーティリティ（名前衝突を解消）
# ==============================

# 1) 取得ターゲット時刻を作る（JST基準・tzなしdatetime）
def build_openmeteo_target_dt(jst_date, race_slot: str):
    h = SESSION_HOUR.get(race_slot, 11)
    if isinstance(jst_date, datetime):
        jst_date = jst_date.date()
    try:
        y, m, d = jst_date.year, jst_date.month, jst_date.day
    except Exception:
        dt = pd.to_datetime(str(jst_date))
        y, m, d = dt.year, dt.month, dt.day
    return datetime(y, m, d, h, 0, 0)




# ==============================
# UI
# ==============================
st.sidebar.header("開催情報 / バンク・風・頭数")
n_cars = st.sidebar.selectbox("出走数（5〜9）", [5,6,7,8,9], index=2)

track_names = list(KEIRIN_DATA.keys())
track = st.sidebar.selectbox(
    "競輪場（プリセット）",
    track_names,
    index=track_names.index("川崎") if "川崎" in track_names else 0
)
info = KEIRIN_DATA[track]
st.session_state["track"] = track

with st.sidebar.expander("📊 会場別 成績補正", expanded=True):
    venue_hit_input = st.text_input(
        "的中率（的中R/投票R）",
        value="",
        placeholder="例：12/40",
        key="venue_hit_input",
    )
    venue_return_input = st.text_input(
        "回収率（払戻/投資）",
        value="",
        placeholder="例：12000/8000",
        key="venue_return_input",
    )

    venue_hit_rate = parse_fraction_rate(venue_hit_input, percent=True)
    venue_return_rate = parse_fraction_rate(venue_return_input, percent=True)
    venue_profile = judge_venue_profile(venue_hit_rate, venue_return_rate)

    venue_home_flow_mult = venue_home_flow_multiplier(track, venue_profile)
    venue_min_odds_mult = float(VENUE_MIN_ODDS_MULT.get(venue_profile, 1.00))
    venue_hit_expect_coef = _venue_fit_hit_coef(venue_hit_rate)
    venue_myoumi_expect_coef = _venue_fit_myoumi_coef(venue_return_rate)

    venue_shape = calc_venue_shape_index(track)

    hit_txt = "—" if venue_hit_rate is None else f"{venue_hit_rate:.1f}%"
    ret_txt = "—" if venue_return_rate is None else f"{venue_return_rate:.1f}%"

    st.write(f"的中率：**{hit_txt}**")
    st.write(f"回収率：**{ret_txt}**")
    st.write(f"会場判定：**{venue_profile}**")
    st.write(f"開催適合補正：的中期待×**{venue_hit_expect_coef:.2f}** ／ 妙味期待×**{venue_myoumi_expect_coef:.2f}**")
    st.write(f"最終H補正倍率：**{venue_home_flow_mult:.2f}**")
    st.write(f"必要オッズ倍率：**{venue_min_odds_mult:.2f}**")
    st.caption(
        f"みなし直線率 {venue_shape.get('minashi_ratio', 0.0):.3f} / "
        f"カント支え {venue_shape.get('bank_support', 0.0):.1f} / "
        f"形状リスク {venue_shape.get('stretch_risk', 0.0):.2f}"
    )

st.session_state["venue_hit_rate"] = venue_hit_rate
st.session_state["venue_return_rate"] = venue_return_rate
st.session_state["venue_profile"] = venue_profile
st.session_state["venue_home_flow_mult"] = venue_home_flow_mult
st.session_state["venue_min_odds_mult"] = venue_min_odds_mult
st.session_state["venue_hit_expect_coef"] = venue_hit_expect_coef
st.session_state["venue_myoumi_expect_coef"] = venue_myoumi_expect_coef

st.sidebar.markdown("### 🏟️ 開催場決まり手成績")
with st.sidebar.expander("数値入力（オッズパーク等の表をそのまま％入力）", expanded=True):
    venue_kimarite_enabled = st.checkbox(
        "決まり手補正を使う",
        value=bool(st.session_state.get("venue_kimarite_enabled", False)),
        key="venue_kimarite_enabled",
    )
    st.caption("オッズパーク等の表をそのまま％で入力。例：13.9 / 62.4 / 24.2")

    c1, c2, c3 = st.columns(3)
    with c1:
        vk_win_escape = st.number_input("1着 逃げ%", 0.0, 100.0, float(st.session_state.get("vk_win_escape", 0.0) or 0.0), 0.1, key="vk_win_escape")
    with c2:
        vk_win_sashi = st.number_input("1着 差し%", 0.0, 100.0, float(st.session_state.get("vk_win_sashi", 0.0) or 0.0), 0.1, key="vk_win_sashi")
    with c3:
        vk_win_makuri = st.number_input("1着 捲り%", 0.0, 100.0, float(st.session_state.get("vk_win_makuri", 0.0) or 0.0), 0.1, key="vk_win_makuri")

    c4, c5, c6, c7 = st.columns(4)
    with c4:
        vk_sec_escape = st.number_input("2着 逃げ%", 0.0, 100.0, float(st.session_state.get("vk_sec_escape", 0.0) or 0.0), 0.1, key="vk_sec_escape")
    with c5:
        vk_sec_sashi = st.number_input("2着 差し%", 0.0, 100.0, float(st.session_state.get("vk_sec_sashi", 0.0) or 0.0), 0.1, key="vk_sec_sashi")
    with c6:
        vk_sec_makuri = st.number_input("2着 捲り%", 0.0, 100.0, float(st.session_state.get("vk_sec_makuri", 0.0) or 0.0), 0.1, key="vk_sec_makuri")
    with c7:
        vk_sec_mark = st.number_input("2着 マーク%", 0.0, 100.0, float(st.session_state.get("vk_sec_mark", 0.0) or 0.0), 0.1, key="vk_sec_mark")

    vk_sample_count = st.number_input(
        "回数",
        min_value=0,
        max_value=10000,
        value=int(st.session_state.get("vk_sample_count", 0) or 0),
        step=1,
        key="vk_sample_count",
    )

    VENUE_KIMARITE_STATS = {
        "enabled": bool(venue_kimarite_enabled),
        "win_escape": float(vk_win_escape),
        "win_sashi": float(vk_win_sashi),
        "win_makuri": float(vk_win_makuri),
        "sec_escape": float(vk_sec_escape),
        "sec_sashi": float(vk_sec_sashi),
        "sec_makuri": float(vk_sec_makuri),
        "sec_mark": float(vk_sec_mark),
        "sample_count": int(vk_sample_count),
    }

    _vk_role_bonus_preview, _vk_rel_preview, _vk_detail_preview = _calc_venue_kimarite_role_bonus_map(VENUE_KIMARITE_STATS)
    st.caption(
        "補正プレビュー："
        f"先頭 {_fmt_signed_pt(_vk_role_bonus_preview.get('head', 0.0))} / "
        f"番手 {_fmt_signed_pt(_vk_role_bonus_preview.get('second', 0.0))} / "
        f"3番手以降 {_fmt_signed_pt(_vk_role_bonus_preview.get('thirdplus', 0.0))} / "
        f"単騎 {_fmt_signed_pt(_vk_role_bonus_preview.get('single', 0.0))} "
        f"｜信頼係数 {_vk_rel_preview:.2f}"
    )

globals()["VENUE_KIMARITE_STATS"] = VENUE_KIMARITE_STATS
st.session_state["VENUE_KIMARITE_STATS"] = VENUE_KIMARITE_STATS

with st.sidebar.expander("🎯 流れ1-2｜下限計算", expanded=False):
    flow_switch_total_races = st.text_input(
        "総レース数",
        value=str(st.session_state.get("flow_switch_total_races", "") or ""),
        key="flow_switch_total_races",
    )
    flow_switch_12_nifuku_hits = st.text_input(
        "1-2的中数",
        value=str(st.session_state.get("flow_switch_12_nifuku_hits", "") or ""),
        key="flow_switch_12_nifuku_hits",
    )
    flow_switch_12_nifuku_under3_races = st.text_input(
        "推奨1-2 2車複 3倍以下対象R数",
        value=str(st.session_state.get("flow_switch_12_nifuku_under3_races", "") or ""),
        key="flow_switch_12_nifuku_under3_races",
    )
    flow_switch_12_wide_hits = st.text_input(
        "推奨1-2ワイド的中数",
        value=str(st.session_state.get("flow_switch_12_wide_hits", "") or ""),
        key="flow_switch_12_wide_hits",
    )
    flow_switch_target_ev = st.text_input(
        "目標EV",
        value=str(st.session_state.get("flow_switch_target_ev", "") or ""),
        key="flow_switch_target_ev",
    )
    flow_switch_safety_factor = st.text_input(
        "安全係数",
        value=str(st.session_state.get("flow_switch_safety_factor", "") or ""),
        key="flow_switch_safety_factor",
    )

    flow_switch_stats = _calc_flow_switch_stats(
        flow_switch_total_races,
        flow_switch_12_wide_hits,
        flow_switch_target_ev,
        flow_switch_safety_factor,
        flow_switch_12_nifuku_under3_races,
        flow_switch_12_nifuku_hits,
    )
    _flow12_floor = flow_switch_stats.get("trio12_all", {}).get("recommended_floor_odds", None)
    _under3_rate = flow_switch_stats.get("nifuku12_under3_rate", None)
    _wide_market_basis = flow_switch_stats.get("wide_market_switch_odds", None)
    _switch3412_floor = flow_switch_stats.get("switch_3412", {}).get("recommended_floor_odds", None)
    st.caption(f"1-2 3倍以下率：{_fmt_pct(_under3_rate)}")
    st.caption(f"1-2市場ワイド切替目安：{_fmt_odds(_wide_market_basis)}")
    st.caption(f"推奨下限合成オッズ：{_fmt_odds(_flow12_floor)}")
    st.caption(f"34-12必要合成オッズ：{_fmt_odds(_switch3412_floor)}")

globals()["FLOW_SWITCH_STATS"] = _get_flow_switch_stats_from_state()
globals()["FLOW_12_ALL_TRIO_SWITCH_ODDS_THRESHOLD"] = _flow_12_all_recommended_floor()

race_time = st.sidebar.selectbox("開催区分", ["モーニング","デイ","ナイター","ミッドナイト"], 1)
race_day = st.sidebar.date_input("日付（風取得用）", value=date.today())

wind_dir = st.sidebar.selectbox(
    "風向", ["無風","左上","上","右上","左","右","左下","下","右下"],
    index=0, key="wind_dir_input"
)

wind_speed_default = st.session_state.get("wind_speed", 3.0)
wind_speed = st.sidebar.number_input("風速(m/s)", 0.0, 60.0, float(wind_speed_default), 0.1)

with st.sidebar.expander("🌀 風をAPIで自動取得（Open-Meteo）", expanded=False):
    st.sidebar.caption("基準時刻：モ=8時 / デ=11時 / ナ=18時 / ミ=22時（JST・tzなしで取得）")

    if st.sidebar.button("APIで取得→風速に反映", use_container_width=True):
        info_xy = VELODROME_MASTER.get(track)
        if not info_xy or info_xy.get("lat") is None or info_xy.get("lon") is None:
            st.sidebar.error(f"{track} の座標が未登録です（VELODROME_MASTER に lat/lon を入れてください）")
        else:
            try:
                target = build_openmeteo_target_dt(race_day, race_time)
                data = fetch_openmeteo_hour(info_xy["lat"], info_xy["lon"], target)

                st.session_state["wind_speed"] = round(float(data["speed_ms"]), 2)

                precip = float(data.get("precipitation", 0.0) or 0.0)
                weather_code = data.get("weather_code", None)

                st.session_state["precipitation"] = precip
                st.session_state["weather_code"] = weather_code
                st.session_state["is_wet"] = bool(precip >= 0.3)

                st.sidebar.success(
                    f"{track} {target:%Y-%m-%d %H:%M} "
                    f"風速 {st.session_state['wind_speed']:.1f} m/s "
                    f"降水 {precip:.1f}mm/h "
                    f"（API側と{data['diff_min']:.0f}分ズレ）"
                )
                st.rerun()

            except Exception as e:
                st.sidebar.error(f"取得に失敗：{e}")



straight_length = st.sidebar.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle      = st.sidebar.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length     = st.sidebar.number_input("周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)
st.session_state["bank_length"] = float(bank_length)

base_laps = st.sidebar.number_input("周回（通常4）", 1, 10, 4, 1)
day_label = st.sidebar.selectbox(
    "開催日",
    ["初日", "2日目", "3日目", "4日目", "5日目", "最終日"],
    0
)

DAY_LAP_ADD = {
    "初日": 1,
    "2日目": 2,
    "3日目": 3,
    "4日目": 4,
    "5日目": 5,
    "最終日": 6,
}

eff_laps = int(base_laps) + DAY_LAP_ADD[day_label]

race_class = st.sidebar.selectbox(
    "級別",
    ["Ｓ級", "Ａ級", "Ａ級チャレンジ", "ガールズ", "アドバンス"],
    0
)

is_girls_like = race_class in ("ガールズ", "アドバンス")

# === 会場styleを「得意会場平均」を基準に再定義
zL, zTH, dC = venue_z_terms(straight_length, bank_angle, bank_length)
style_raw = venue_mix(zL, zTH, dC)

# 天候による自動バイアス補正
precip = float(st.session_state.get("precipitation", 0.0) or 0.0)

if precip >= 5.0:
    weather_override = 0.6
elif precip >= 2.0:
    weather_override = 0.4
elif precip >= 0.3:
    weather_override = 0.2
else:
    weather_override = 0.0

manual_override = st.sidebar.slider(
    "会場バイアス補正（−2差し ←→ +2先行）",
    -2.0, 2.0, 0.0, 0.1
)

override = clamp(manual_override + weather_override, -2.0, 2.0)

st.sidebar.caption(
    f"天候自動補正：{weather_override:+.1f} / 最終バイアス補正：{override:+.1f}"
)

style = clamp(style_raw + 0.25 * override, -1.0, +1.0)



CLASS_FACTORS = {
    "Ｓ級":           {"spread":1.00, "line":1.00},
    "Ａ級":           {"spread":0.90, "line":0.85},
    "Ａ級チャレンジ": {"spread":0.80, "line":0.70},
    "ガールズ":       {"spread":0.85, "line":1.00},
    "アドバンス":     {"spread":0.85, "line":1.00},
}
cf = CLASS_FACTORS[race_class]

DAY_FACTOR = {
    "初日": 1.00,
    "2日目": 1.00,
    "3日目": 0.99,
    "4日目": 0.98,
    "5日目": 0.97,
    "最終日": 0.96,
}
day_factor = DAY_FACTOR[day_label]

cap_base = clamp(0.06 + 0.02*style, 0.04, 0.08)
line_factor_eff = cf["line"] * day_factor
cap_SB_eff = cap_base * day_factor
if race_time == "ミッドナイト":
    line_factor_eff *= 0.95
    cap_SB_eff *= 0.95

# ===== 日程・級別・頭数で“周回疲労の効き”を薄くシフト（出力には出さない） =====
DAY_SHIFT = {
    "初日": -0.5,
    "2日目": 0.0,
    "3日目": +0.2,
    "4日目": +0.4,
    "5日目": +0.6,
    "最終日": +0.8,
}
CLASS_SHIFT = {
    "Ｓ級": 0.0,
    "Ａ級": +0.10,
    "Ａ級チャレンジ": +0.20,
    "ガールズ": -0.10,
    "アドバンス": -0.10,
}
HEADCOUNT_SHIFT = {5: -0.20, 6: -0.10, 7: -0.05, 8: 0.0, 9: +0.10}

def fatigue_extra(eff_laps: int, day_label: str, n_cars: int, race_class: str) -> float:
    d = float(DAY_SHIFT.get(day_label, 0.0))
    c = float(CLASS_SHIFT.get(race_class, 0.0))
    h = float(HEADCOUNT_SHIFT.get(int(n_cars), 0.0))
    x = (float(eff_laps) - 2.0) + d + c + h
    return max(0.0, x)

# === PATCH-L200:（以下そのまま） ==========================================
# ...（あなたの last200_bonus 以降は変更なし）

fatigue_value = fatigue_extra(eff_laps, day_label, n_cars, race_class)

globals()["fatigue_value"] = float(fatigue_value)
globals()["fatigue_extra_value"] = float(fatigue_value)

# sidebarの直後あたり（straight_length/style/wind_speedが確定した後）
globals()["straight_length"] = float(straight_length)
globals()["bank_length"]     = float(bank_length)
globals()["bank_angle"]      = float(bank_angle)
globals()["style"]           = float(style)
globals()["wind_speed"]      = float(wind_speed)
globals()["race_class"]      = str(race_class)
globals()["venue_profile"]   = str(st.session_state.get("venue_profile", "unknown"))
globals()["venue_home_flow_mult"] = float(st.session_state.get("venue_home_flow_mult", 1.00))
globals()["venue_min_odds_mult"]  = float(st.session_state.get("venue_min_odds_mult", 1.00))
globals()["venue_hit_expect_coef"] = float(st.session_state.get("venue_hit_expect_coef", 1.00))
globals()["venue_myoumi_expect_coef"] = float(st.session_state.get("venue_myoumi_expect_coef", 1.00))
globals()["n_cars"]          = int(n_cars)
globals()["day_label"] = str(day_label)
globals()["eff_laps"]  = int(eff_laps)
    


# ==============================
# メイン：入力
# ==============================
st.title("⭐ ヴェロビ（級別×日程ダイナミクス / 5〜9車・買い目付き：統合版）⭐")
st.caption(f"風補正モード: {WIND_MODE}（'speed_only'=風速のみ / 'directional'=向きも薄く考慮）")

# ←★ここに貼る（1回だけ走らせる）
if "_DID_SELF_GREP" not in st.session_state:
    st.session_state["_DID_SELF_GREP"] = True
    _grep_self("KO使用スコア", __file__, context=6)
    _grep_self("KO使用スコア（降順）", __file__, context=6)
    _grep_self("ko_text", __file__, context=6)
# →★ここまで


st.subheader("2026/05/24更新")
if "race_no_main" not in st.session_state:
    st.session_state["race_no_main"] = 1
c1, c2, c3 = st.columns([6,2,2])
with c1:
    race_no_input = st.number_input("R", min_value=1, max_value=12, step=1,
                                    value=int(st.session_state["race_no_main"]),
                                    key="race_no_input")
with c2:
    prev_clicked = st.button("◀ 前のR", use_container_width=True)
with c3:
    next_clicked = st.button("次のR ▶", use_container_width=True)
if prev_clicked:
    st.session_state["race_no_main"] = max(1, int(race_no_input) - 1); st.rerun()
elif next_clicked:
    st.session_state["race_no_main"] = min(12, int(race_no_input) + 1); st.rerun()
else:
    st.session_state["race_no_main"] = int(race_no_input)
race_no = int(st.session_state["race_no_main"])

# ==============================
# メイン入力：通常入力 → 反映ボタンで計算用データを固定
# ※スコア計算ロジックは元コードから変更しない
# ==============================

# ライン構成（最大7：単騎も1ライン）
line_inputs_live = [
    st.text_input("ライン1（例：123）", key="line_1", max_chars=9),
    st.text_input("ライン2（例：456）", key="line_2", max_chars=9),
    st.text_input("ライン3（例：789）", key="line_3", max_chars=9),
    st.text_input("ライン4（任意）", key="line_4", max_chars=9),
    st.text_input("ライン5（任意）", key="line_5", max_chars=9),
    st.text_input("ライン6（任意）", key="line_6", max_chars=9),
    st.text_input("ライン7（任意）", key="line_7", max_chars=9),
    st.text_input("ライン8（任意）", key="line_8", max_chars=9),
    st.text_input("ライン9（任意）", key="line_9", max_chars=9),
]
n_cars = int(n_cars)
lines_live = [extract_car_list(x, n_cars) for x in line_inputs_live if str(x).strip()]
line_def_live, car_to_group_live = build_line_maps(lines_live)
active_cars_live = sorted({c for lst in lines_live for c in lst}) if lines_live else list(range(1, n_cars+1))

# v179：単騎ラインも含めた認識確認
if lines_live:
    st.caption(
        f"ライン認識：{_format_lines_for_check(lines_live)} "
        f"｜入力済み車番：{''.join(str(x) for x in active_cars_live)} "
        f"（{len(active_cars_live)}/{int(n_cars)}車）"
    )

# 5〜9車対応：ライン入力漏れチェック（単騎も1車としてカウント）
if len(active_cars_live) != int(n_cars):
    st.warning(
        f"出走数{n_cars}に対して、ライン入力済みは{len(active_cars_live)}車です。"
        " ライン入力漏れを確認してください。"
    )

# -----------------------------------------
# 市場印入力（計算反映前）
# ※全体妙味・2列目繰り上げ・フォメ生成に使うため、反映ボタンより前に置く
# ※出走表を見たまま入力できるように「車番ごとに印を選ぶ」形式にする
# ※内部では従来通り market_honmei_raw / market_taikou_raw / market_tan_raw / market_batsu_raw に変換する
# -----------------------------------------
st.caption("市場印入力（計算反映前）")
st.caption("各車番ごとに外部印を選択してください（未選択は —）。")

_market_mark_options_live = ["—", "◎", "〇", "△", "×"]

# 旧UIで選んでいた値が残っている場合は、初期表示に引き継ぐ
_old_mark_by_car_live = {}
_old_pairs_live = [
    ("◎", st.session_state.get(f"market_honmei_car_r{race_no}", "—")),
    ("〇", st.session_state.get(f"market_taikou_car_r{race_no}", "—")),
    ("△", st.session_state.get(f"market_tan_car_r{race_no}", "—")),
    ("×", st.session_state.get(f"market_batsu_car_r{race_no}", "—")),
]
for _mk, _car in _old_pairs_live:
    if str(_car) != "—":
        _old_mark_by_car_live[str(_car)] = _mk

market_mark_by_car_live = {}

# 1行目：見出し
_header_cols = st.columns([0.9, 1, 1, 1, 1, 1])
_header_cols[0].markdown("**車番**")
for _i, _label in enumerate(_market_mark_options_live, start=1):
    _header_cols[_i].markdown(f"**{_label}**")

# 車番ごとに印を選択
for no in sorted(active_cars_live):
    no_str = str(no)
    default_mark = _old_mark_by_car_live.get(no_str, "—")
    default_idx = _market_mark_options_live.index(default_mark) if default_mark in _market_mark_options_live else 0

    row_cols = st.columns([0.9, 5])
    row_cols[0].markdown(f"**{no}番**")
    with row_cols[1]:
        market_mark_by_car_live[no] = st.radio(
            f"{no}番の市場印",
            _market_mark_options_live,
            index=default_idx,
            horizontal=True,
            key=f"market_mark_by_car_r{race_no}_{no}",
            label_visibility="collapsed",
        )

# 車番→印を、従来形式（印→車番）へ変換
_mark_to_cars_live = {"◎": [], "〇": [], "△": [], "×": []}
for no in sorted(active_cars_live):
    mk = market_mark_by_car_live.get(no, "—")
    if mk in _mark_to_cars_live:
        _mark_to_cars_live[mk].append(str(no))

_duplicate_marks_live = [mk for mk, cars in _mark_to_cars_live.items() if len(cars) >= 2]
if _duplicate_marks_live:
    st.warning(
        "同じ印が複数の車番に入っています。"
        "各印は1車だけにしてください。計算上は車番昇順で先頭の車を採用します。"
    )

market_honmei_raw_live = _mark_to_cars_live["◎"][0] if _mark_to_cars_live["◎"] else "—"
market_taikou_raw_live = _mark_to_cars_live["〇"][0] if _mark_to_cars_live["〇"] else "—"
market_tan_raw_live    = _mark_to_cars_live["△"][0] if _mark_to_cars_live["△"] else "—"
market_batsu_raw_live  = _mark_to_cars_live["×"][0] if _mark_to_cars_live["×"] else "—"

_market_selected_live = [
    ("◎", market_honmei_raw_live),
    ("〇", market_taikou_raw_live),
    ("△", market_tan_raw_live),
    ("×", market_batsu_raw_live),
]
_market_summary_live = "　".join(
    f"{mk}{car}" for mk, car in _market_selected_live if str(car) != "—"
)
st.caption(f"入力印：{_market_summary_live if _market_summary_live else 'なし'}")

# ←←← ここに入れる
def input_float_text(label: str, key: str, placeholder: str = ""):
    s = st.text_input(label, value=st.session_state.get(key, ""), key=key, placeholder=placeholder)
    ss = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if ss == "":
        return None
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", ss):
        st.warning(f"{label} は数値で入力してください（入力値: {s}）")
        return None
    return float(ss)

# →→→ ここまで

st.subheader("個人データ（直近4か月：回数）")
cols = st.columns(len(active_cars_live))
ratings_live, S_live, H_live, B_live = {}, {}, {}, {}

k_esc_live, k_mak_live, k_sashi_live, k_mark_live = {}, {}, {}, {}
x1_live, x2_live, x3_live, x_out_live = {}, {}, {}, {}

for i, no in enumerate(active_cars_live):
    with cols[i]:
        st.markdown(f"**{no}番**")
        ratings_live[no] = input_float_text("得点（空欄可）", key=f"pt_{no}", placeholder="例: 55.0")
        S_live[no] = st.number_input("S", 0, 99, 0, key=f"s_{no}")
        H_live[no] = st.number_input("H", 0, 99, 0, key=f"h_{no}")
        B_live[no] = st.number_input("B", 0, 99, 0, key=f"b_{no}")
        k_esc_live[no]   = st.number_input("逃", 0, 99, 0, key=f"ke_{no}")
        k_mak_live[no]   = st.number_input("捲", 0, 99, 0, key=f"km_{no}")
        k_sashi_live[no] = st.number_input("差", 0, 99, 0, key=f"ks_{no}")
        k_mark_live[no]  = st.number_input("マ", 0, 99, 0, key=f"kk_{no}")
        x1_live[no]  = st.number_input("1着", 0, 99, 0, key=f"x1_{no}")
        x2_live[no]  = st.number_input("2着", 0, 99, 0, key=f"x2_{no}")
        x3_live[no]  = st.number_input("3着", 0, 99, 0, key=f"x3_{no}")
        x_out_live[no]= st.number_input("着外", 0, 99, 0, key=f"xo_{no}")

# =====================================================
# コメントチェック表
#   前検コメントを見て手動チェック
#   自力：自力 / 自力基本 / 自分で / 前で 等
#   自力自在：自力自在 / 何でもやる / 前々自力 等
#   自在：自在 / 前々 / 流れで / 位置取り 等
#   番手：○○君 / ○○へ / 任せる / 近畿勢 等
#   単騎：一人で / 単騎で / 決めず 等（ライン入力上の単騎とは別のコメント補助）
#   競り：競り対象の車番にチェックし、競り相手を選択
#   後位信頼：3番手以降の明確追走/地区まとめ/流動を手動評価
# =====================================================
st.subheader("コメントチェック")

jiryoku_comment_live = {}
jiryoku_jizai_comment_live = {}
jizai_comment_live = {}
target_comment_live = {}
single_comment_live = {}
seri_comment_live = {}
seri_target_live = {}
line_follow_trust_live = {}

comment_cols = st.columns(len(active_cars_live))

for i, no in enumerate(active_cars_live):
    no = int(no)
    with comment_cols[i]:
        st.markdown(f"**{no}番**")

        jiryoku_comment_live[no] = st.checkbox(
            "自力",
            value=False,
            key=f"jiryoku_comment_r{race_no}_{no}"
        )

        jiryoku_jizai_comment_live[no] = st.checkbox(
            "自力自在",
            value=False,
            key=f"jiryoku_jizai_comment_r{race_no}_{no}"
        )

        jizai_comment_live[no] = st.checkbox(
            "自在",
            value=False,
            key=f"jizai_comment_r{race_no}_{no}"
        )

        target_comment_live[no] = st.checkbox(
            "番手",
            value=False,
            key=f"target_comment_r{race_no}_{no}"
        )

        single_comment_live[no] = st.checkbox(
            "単騎",
            value=False,
            key=f"single_comment_r{race_no}_{no}"
        )

        seri_comment_live[no] = st.checkbox(
            "競り",
            value=False,
            key=f"seri_comment_r{race_no}_{no}"
        )

        _seri_target_options = ["—"] + [int(x) for x in active_cars_live if int(x) != int(no)]
        _seri_target_sel = st.selectbox(
            "競り相手",
            options=_seri_target_options,
            index=0,
            key=f"seri_target_r{race_no}_{no}"
        )
        seri_target_live[no] = None if _seri_target_sel == "—" else int(_seri_target_sel)

        # v125: 後位信頼はselectboxではなくチェックボックス式。
        # 単騎コメントは後位信頼ではなく、上の「単騎」チェックで独立管理する。
        # 複数チェック時は、リスクが強い順に 流動 > 地区まとめ > 明確追走 で採用する。
        _old_line_follow_key = f"line_follow_trust_r{race_no}_{no}"
        _old_line_follow_val = str(st.session_state.get(_old_line_follow_key, "通常") or "通常")

        st.caption("後位信頼")
        _lft_clear = st.checkbox(
            "明確",
            value=(_old_line_follow_val == "明確追走"),
            key=f"line_follow_clear_r{race_no}_{no}"
        )
        _lft_district = st.checkbox(
            "地区",
            value=(_old_line_follow_val == "地区まとめ"),
            key=f"line_follow_district_r{race_no}_{no}"
        )
        _lft_flow = st.checkbox(
            "流動",
            value=(_old_line_follow_val == "流動"),
            key=f"line_follow_flow_r{race_no}_{no}"
        )

        _lft_checked_count = sum([
            bool(_lft_clear),
            bool(_lft_district),
            bool(_lft_flow),
        ])
        if _lft_checked_count >= 2:
            st.caption("※複数時は強リスク側を採用")

        if _lft_flow:
            line_follow_trust_live[no] = "流動"
        elif _lft_district:
            line_follow_trust_live[no] = "地区まとめ"
        elif _lft_clear:
            line_follow_trust_live[no] = "明確追走"
        else:
            line_follow_trust_live[no] = "通常"

st.markdown("---")

apply_input = st.button(
    "入力を反映して計算する",
    type="primary",
    use_container_width=True,
    key="apply_input_main"
)

if apply_input:
    st.session_state["race_snapshot"] = {
        "line_inputs": list(line_inputs_live),
        "lines": [list(x) for x in lines_live],
        "line_def": {g: list(mem) for g, mem in line_def_live.items()},
        "car_to_group": dict(car_to_group_live),
        "active_cars": list(active_cars_live),

        "market_honmei_raw": market_honmei_raw_live,
        "market_taikou_raw": market_taikou_raw_live,
        "market_tan_raw": market_tan_raw_live,
        "market_batsu_raw": market_batsu_raw_live,
        # v20: 車番ごとの外部印をそのまま保存する。
        # ここを保存しないと、後段で印→車番の圧縮値から復元するため、
        # 表示上の車番と印がズレる原因になる。
        "market_mark_by_car": {int(k): str(v) for k, v in market_mark_by_car_live.items()},

        "ratings": dict(ratings_live),
        "S": dict(S_live),
        "H": dict(H_live),
        "B": dict(B_live),

        "k_esc": dict(k_esc_live),
        "k_mak": dict(k_mak_live),
        "k_sashi": dict(k_sashi_live),
        "k_mark": dict(k_mark_live),

        "x1": dict(x1_live),
        "x2": dict(x2_live),
        "x3": dict(x3_live),
        "x_out": dict(x_out_live),

        "jiryoku_comment": dict(jiryoku_comment_live),
        "jiryoku_jizai_comment": dict(jiryoku_jizai_comment_live),
        "jizai_comment": dict(jizai_comment_live),
        "target_comment": dict(target_comment_live),
        "single_comment": dict(single_comment_live),
        "seri_comment": dict(seri_comment_live),
        "seri_target": dict(seri_target_live),
        "line_follow_trust": dict(line_follow_trust_live),
    }

snapshot = st.session_state.get("race_snapshot")

if snapshot is None:
    st.info("入力後、『入力を反映して計算する』を押すと本計算します。")
    st.stop()

# ==============================
# ここから下は、反映済みデータだけで計算する
# ==============================

line_inputs = snapshot["line_inputs"]
lines = snapshot["lines"]
line_def = snapshot["line_def"]
car_to_group = snapshot["car_to_group"]
active_cars = snapshot["active_cars"]

ratings = snapshot["ratings"]
S = snapshot["S"]
H = snapshot["H"]
B = snapshot["B"]

k_esc = snapshot["k_esc"]
k_mak = snapshot["k_mak"]
k_sashi = snapshot["k_sashi"]
k_mark = snapshot["k_mark"]

x1 = snapshot["x1"]
x2 = snapshot["x2"]
x3 = snapshot["x3"]
x_out = snapshot["x_out"]

jiryoku_comment = snapshot.get("jiryoku_comment", {})
jiryoku_jizai_comment = snapshot.get("jiryoku_jizai_comment", {})
jizai_comment = snapshot.get("jizai_comment", {})
target_comment = snapshot.get("target_comment", {})
single_comment = snapshot.get("single_comment", {})
seri_comment = snapshot.get("seri_comment", {})
seri_target = snapshot.get("seri_target", {})
line_follow_trust = snapshot.get("line_follow_trust", {})

globals()["jiryoku_comment"] = jiryoku_comment
globals()["jiryoku_jizai_comment"] = jiryoku_jizai_comment
globals()["jizai_comment"] = jizai_comment
globals()["target_comment"] = target_comment
globals()["single_comment"] = single_comment
globals()["seri_comment"] = seri_comment
globals()["seri_target"] = seri_target
globals()["line_follow_trust"] = line_follow_trust

st.caption(
    "反映済みデータで計算中："
    f"車番={active_cars} ／ "
    f"ライン={'　'.join(''.join(map(str, ln)) for ln in lines) if lines else 'なし'}"
)

# 反映済みデータの整合チェック
if len(active_cars) != int(n_cars):
    st.error(
        f"出走数{n_cars}に対して、反映済みラインは{len(active_cars)}車です。"
        f" 反映済み車番: {active_cars}"
    )
    st.stop()

dup_check = []
for lst in lines:
    dup_check.extend(lst)

dups = sorted([x for x in set(dup_check) if dup_check.count(x) >= 2])

if dups:
    st.error(f"同じ車番が複数ラインに入っています: {dups}")
    st.stop()

ratings_val = {no: (float(ratings[no]) if ratings[no] is not None else 55.0) for no in active_cars}

# =====================================================
# 混戦度判定：競走得点1位と2位の差
# ※ active_cars / ratings_val が確定した後で実行する
# =====================================================
race_compact = calc_race_compactness(ratings_val, active_cars)
race_compact_label = race_compact.get("label", "未判定")
race_compact_gap = race_compact.get("top_gap", None)

globals()["race_compact_label"] = race_compact_label
globals()["race_compact_gap"] = race_compact_gap
globals()["race_compact"] = race_compact

# H：最終ホーム想定ライン
home_line_scores = calc_home_line_scores(line_def, H, B, active_cars)

# H：最終ホーム想定ライン
home_line_scores = calc_home_line_scores(line_def, H, B, active_cars)
home_line_order = make_home_line_order(line_def, H, B, active_cars)
home_line_text = format_home_line_order(line_def, home_line_order)

home_top_gid = home_line_order[0] if home_line_order else None
home_second_gid = home_line_order[1] if len(home_line_order) >= 2 else None
globals()["home_top_gid"] = home_top_gid
globals()["home_second_gid"] = home_second_gid

# H主導ライン判定
# Hスコアが低すぎる場合は「主導なし」とする
home_top_score = float(home_line_scores.get(home_top_gid, 0.0)) if home_top_gid is not None else 0.0

if home_top_gid is not None and home_top_score >= 1.0:
    home_top_line = format_home_line_order(line_def, [home_top_gid])
else:
    home_top_line = "主導なし"



# 1着・2着の縮約（級別×会場の事前分布を混ぜる）
def prior_by_class(cls, style_adj):
    if "ガール" in cls: p1,p2 = 0.18,0.24
    elif "Ｓ級" in cls: p1,p2 = 0.22,0.26
    elif "チャレンジ" in cls: p1,p2 = 0.18,0.22
    else: p1,p2 = 0.20,0.25
    p1 += 0.010*style_adj; p2 -= 0.005*style_adj
    return clamp(p1,0.05,0.60), clamp(p2,0.05,0.60)

def n0_by_n(n):
    if n<=6: return 12
    if n<=14: return 8
    if n<=29: return 5
    return 3

# === 1〜3着＋着外を “ちゃんと” Form に反映する版（ここだけ置換） ===
p1_eff, p2_eff, p3_eff, pout_eff = {}, {}, {}, {}

for no in active_cars:
    n = x1[no] + x2[no] + x3[no] + x_out[no]

    # 既存：クラス×脚質の prior（あなたの関数をそのまま使う）
    p1_prior, p2_prior = prior_by_class(race_class, style)

    # 追加：3着＆着外の prior（まずは固定で安全運用）
    p3_prior   = 0.10
    pout_prior = 0.55

    n0 = n0_by_n(n)

    if n == 0:
        p1_eff[no], p2_eff[no] = p1_prior, p2_prior
        p3_eff[no]             = p3_prior
        pout_eff[no]           = pout_prior
    else:
        p1_eff[no]  = clamp((x1[no]    + n0*p1_prior ) / (n + n0), 0.0, 0.40)
        p2_eff[no]  = clamp((x2[no]    + n0*p2_prior ) / (n + n0), 0.0, 0.50)
        p3_eff[no]  = clamp((x3[no]    + n0*p3_prior ) / (n + n0), 0.0, 0.55)
        pout_eff[no]= clamp((x_out[no] + n0*pout_prior) / (n + n0), 0.0, 0.95)

    # 合計が暴れない安全弁（1-3着を優先して整える）
    s123 = p1_eff[no] + p2_eff[no] + p3_eff[no]
    if s123 > 0.95:
        scale = 0.95 / s123
        p1_eff[no] *= scale
        p2_eff[no] *= scale
        p3_eff[no] *= scale

    pout_eff[no] = clamp(1.0 - (p1_eff[no] + p2_eff[no] + p3_eff[no]), 0.0, 0.95)

# ★Form：1〜3着を評価、着外は減点（ここが効く）
Form = {
    no: (3.0*p1_eff[no] + 2.0*p2_eff[no] + 1.0*p3_eff[no] - 1.2*pout_eff[no])
    for no in active_cars
}

# === Form 偏差値化（平均50, SD10）
form_list = [Form[n] for n in active_cars]
form_T, mu_form, sd_form, _ = t_score_from_finite(np.array(form_list))
form_T_map = {n: float(form_T[i]) for i, n in enumerate(active_cars)}


# --- 脚質プロフィール（会場適性：得意会場平均基準のstyleを掛ける）
prof_base, prof_escape, prof_sashi, prof_oikomi = {}, {}, {}, {}
for no in active_cars:
    tot = k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no]
    if tot==0: esc=mak=sashi=mark = 0.25
    else:
        esc=k_esc[no]/tot; mak=k_mak[no]/tot; sashi=k_sashi[no]/tot; mark=k_mark[no]/tot
    prof_escape[no]=esc; prof_sashi[no]=sashi; prof_oikomi[no]=mark
    base = esc*BASE_BY_KAKU["逃"] + mak*BASE_BY_KAKU["捲"] + sashi*BASE_BY_KAKU["差"] + mark*BASE_BY_KAKU["マ"]
    vmix = style
    venue_bonus = 0.06 * vmix * ( +1.00*esc + 0.40*mak - 0.60*sashi - 0.25*mark )
    prof_base[no] = base + clamp(venue_bonus, -0.06, +0.06)

# ==============================
# level_rating_scale 保険定義
# ==============================
if "level_rating_scale" not in globals():
    level_rating_scale = 1.0

# ======== 個人補正（得点/脚質上位/着順分布） ========
ratings_sorted = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank = {no: i+1 for i,no in enumerate(ratings_sorted)}
def tenscore_bonus(no):
    r = ratings_rank[no]
    top_n = min(3, len(active_cars))
    bottom_n = min(3, len(active_cars))
    if r <= top_n: return +0.03
    if r >= len(active_cars)-bottom_n+1: return -0.02
    return 0.0
def topk_bonus(k_dict, topn=3, val=0.02):
    order = sorted(k_dict.items(), key=lambda x:(x[1], -x[0]), reverse=True)
    grant = set([no for i,(no,v) in enumerate(order) if i<topn])
    return {no:(val if no in grant else 0.0) for no in k_dict}
esc_bonus   = topk_bonus(k_esc,   topn=3, val=0.02)
mak_bonus   = topk_bonus(k_mak,   topn=3, val=0.02)
sashi_bonus = topk_bonus(k_sashi, topn=3, val=0.015)
mark_bonus  = topk_bonus(k_mark,  topn=3, val=0.01)
def finish_bonus(no):
    tot = x1[no]+x2[no]+x3[no]+x_out[no]
    if tot == 0: return 0.0
    in3 = (x1[no]+x2[no]+x3[no]) / tot
    out = x_out[no] / tot
    bonus = 0.0
    if in3 > 0.50: bonus += 0.03
    if out > 0.70: bonus -= 0.03
    if out < 0.40: bonus += 0.02
    return bonus
extra_bonus = {}
for no in active_cars:
    total = (tenscore_bonus(no) +
             esc_bonus.get(no,0.0) + mak_bonus.get(no,0.0) +
             sashi_bonus.get(no,0.0) + mark_bonus.get(no,0.0) +
             finish_bonus(no))
    extra_bonus[no] = clamp(total, -0.10, +0.10)

# ===== 会場個性を“個人スコア”に浸透：bank系補正（差し替え案） =====

def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi, bank_length=None):
    pe = float(prof_escape or 0.0)
    ps = float(prof_sashi  or 0.0)

    # bank_lengthが渡っていない場合の扱いを決める（例：0.0扱い or venue既定値）
    bl = float(bank_length or 0.0)

    zL, zTH, dC = venue_z_terms(straight_length, bank_angle, bl)

    base = clamp(0.06*zTH - 0.05*zL - 0.03*dC, -0.08, +0.08)
    out  = base * pe - 0.5 * base * ps
    return round(out, 3)


def bank_length_adjust(bank_length, prof_oikomi):
    po = float(prof_oikomi or 0.0)
    L  = float(bank_length or 0.0)
    dC = (+0.4 if L >= 480 else 0.0 if L >= 380 else -0.4)

    out = 0.03 * (-dC) * po
    return round(out, 3)



# --- 安定度（着順分布）をT本体に入れるための重み（強化版） ---
STAB_W_IN3  = 0.18   # 3着内の寄与
STAB_W_OUT  = 0.22   # 着外のペナルティ
STAB_W_LOWN = 0.06   # サンプル不足ペナルティ
STAB_PRIOR_IN3 = 0.33
STAB_PRIOR_OUT = 0.45

def stability_score(no: int) -> float:
    n1 = x1.get(no, 0); n2 = x2.get(no, 0); n3 = x3.get(no, 0); nOut = x_out.get(no, 0)
    n  = n1 + n2 + n3 + nOut
    if n <= 0:
        return 0.0
    # 少サンプル縮約（この関数内で完結）
    if n <= 6:    n0 = 12
    elif n <= 14: n0 = 8
    elif n <= 29: n0 = 5
    else:         n0 = 3

    in3  = (n1 + n2 + n3 + n0*STAB_PRIOR_IN3) / (n + n0)
    out_ = (nOut          + n0*STAB_PRIOR_OUT) / (n + n0)

    bonus = 0.0
    bonus += STAB_W_IN3 * (in3 - STAB_PRIOR_IN3) * 2.0
    bonus -= STAB_W_OUT * (out_ - STAB_PRIOR_OUT) * 2.0

    if n < 10:
        bonus -= STAB_W_LOWN * (10 - n) / 10.0

    # キャップ：nに応じて段階的に広げる（±0.35〜±0.45）
    cap = 0.35
    if n >= 15: cap = 0.45
    elif n >= 10: cap = 0.40

    return clamp(bonus, -cap, +cap)

# ===== SBなし合計（環境補正 + 得点微補正 + 個人補正 + 周回疲労 + 安定度） =====
tens_list = [ratings_val[no] for no in active_cars]
t_corr = tenscore_correction(tens_list) if active_cars else []
tens_corr = {no:t_corr[i] for i,no in enumerate(active_cars)} if active_cars else {}


# ==============================
# L200_RAW（観測用）を先に作る：ここでは laps_adj 等は一切計算しない
# ==============================
_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir",   wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

L200_RAW = {}
for no in active_cars:
    role = role_in_line(no, line_def)

    # --- L200（残脚）生値を計算：ENV合計には“入れない”観測用 ---
    l200 = l200_adjust(
        role=role,
        straight_length=straight_length,
        bank_length=bank_length,
        race_class=race_class,
        prof_escape=float(prof_escape[no]),
        prof_sashi=float(prof_sashi[no]),
        prof_oikomi=float(prof_oikomi[no]),
        is_wet=st.session_state.get("is_wet", False)  # 雨トグル未実装なら False のまま
    )
    L200_RAW[int(no)] = float(l200)


# ==============================
# rows（本体計算）ここで laps_adj を計算して使う（2重計算しない）
# ==============================
rows = []

# H：最終ホーム地力補正マップ
H_Z = calc_h_score_map(H, active_cars)

_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir", wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

# =====================================================
# コメント補正用：競り相手・後位信頼の前処理
# =====================================================
jiryoku_comment_map = globals().get("jiryoku_comment", {}) or {}
jiryoku_jizai_comment_map = globals().get("jiryoku_jizai_comment", {}) or {}
jizai_comment_map   = globals().get("jizai_comment", {}) or {}
target_comment_map  = globals().get("target_comment", {}) or {}
single_comment_map  = globals().get("single_comment", {}) or {}
seri_comment_map    = globals().get("seri_comment", {}) or {}
seri_target_map     = globals().get("seri_target", {}) or {}
line_follow_trust_map = globals().get("line_follow_trust", {}) or {}

seri_incoming_map = {}
try:
    for _src, _dst in (seri_target_map or {}).items():
        try:
            _s = int(_src)
            if _dst is None or str(_dst).strip() in ("", "None", "—"):
                continue
            _d = int(_dst)
            if _s == _d:
                continue
            seri_incoming_map.setdefault(_d, []).append(_s)
        except Exception:
            continue
except Exception:
    seri_incoming_map = {}

def _line_follow_trust_bonus_for_car(_no, _role, _is_girls_like=False):
    """
    3番手以降の追走信頼補正。
    ・「〇〇君へ」等の明確追走は3着内・ライン決着を少し救う。
    ・「関東勢へ」等の地区まとめや「流動」は、裏切り/切替リスクとして減点。
    ・3番手以降だけに効かせ、番手評価を歪ませない。
    """
    try:
        if str(_role) != "thirdplus":
            return 0.0
        label = str(line_follow_trust_map.get(int(_no), "通常") or "通常")
        mp = {
            "明確追走": 0.050,
            "通常": 0.000,
            "地区まとめ": -0.025,
            "流動": -0.080,
            "単騎寄り": -0.120,
        }
        v = float(mp.get(label, 0.0))
        if _is_girls_like:
            v *= 0.50
        return round(clamp(v, -0.120, 0.050), 3)
    except Exception:
        return 0.0

for no in active_cars:
    no = int(no)
    role = role_in_line(no, line_def)

    # =====================================================
    # 周回疲労（DAY×頭数×級別を反映）
    # =====================================================
    extra = fatigue_extra(eff_laps, day_label, n_cars, race_class)
    extra = min(extra, 3.0)   # 応急上限（暴走止め）

    fatigue_scale = (
        1.0  if race_class == "Ｓ級" else
        1.1  if race_class == "Ａ級" else
        1.2  if race_class == "Ａ級チャレンジ" else
        1.05
    )

    # =====================================================
    # 周回疲労補正
    # =====================================================
    laps_adj = (
        -0.10 * extra * (1.0 if float(prof_escape[no]) > 0.5 else 0.0)
        + 0.05 * extra * (1.0 if float(prof_oikomi[no]) > 0.4 else 0.0)
    ) * fatigue_scale

    # ガールズは周回疲労を弱める
    if is_girls_like:
        laps_adj *= 0.3

    # 周回疲労の暴走防止
    laps_adj = clamp(laps_adj, -0.22, 0.18)

    # =====================================================
    # コメント補正
    #   自力：本人をプラス補正
    #   番手：本人ではなく、前の自力先頭をライン連動で格上げ
    #   競り：競り対象者を減点
    # =====================================================
    is_jiryoku_comment = bool(jiryoku_comment_map.get(int(no), False))
    is_jiryoku_jizai_comment = bool(jiryoku_jizai_comment_map.get(int(no), False))
    is_jizai_comment   = bool(jizai_comment_map.get(int(no), False))
    is_single_comment  = bool(single_comment_map.get(int(no), False))
    is_seri_comment    = bool(seri_comment_map.get(int(no), False))
    seri_opponents = []
    try:
        _sel_target = seri_target_map.get(int(no), None)
        if _sel_target is not None and str(_sel_target).strip() not in ("", "None", "—"):
            seri_opponents.append(int(_sel_target))
    except Exception:
        pass
    try:
        seri_opponents.extend([int(x) for x in seri_incoming_map.get(int(no), [])])
    except Exception:
        pass
    seri_opponents = [int(x) for x in dict.fromkeys(seri_opponents) if int(x) != int(no)]

        # -----------------------------------------------------
    # 自力・自力自在・自在コメント補正
    #   3つは原則どれか1つ。
    #   自力自在チェック、または自力＋自在の同時チェックは内部的に「自力自在」として扱う。
    #   大きく順位を作り替えず、軸判定・ステップ判定の補助に留める。
    # -----------------------------------------------------
    if is_jiryoku_jizai_comment or (is_jiryoku_comment and is_jizai_comment):
        move_style = "自力自在"
    elif is_jiryoku_comment:
        move_style = "自力"
    elif is_jizai_comment:
        move_style = "自在"
    else:
        move_style = ""

    jiryoku_comment_bonus = 0.0
    jizai_comment_bonus = 0.0

    if move_style == "自力":
        # 主導力寄り。旧自力補正より少し抑え、コメントだけで順位が動きすぎないようにする。
        jiryoku_comment_bonus = 0.105
        if role == "head":
            jiryoku_comment_bonus += 0.015
        try:
            h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
            if h_line and int(h_line[0]) == int(no):
                jiryoku_comment_bonus += 0.025
        except Exception:
            pass
        if is_girls_like:
            jiryoku_comment_bonus *= 0.60

    elif move_style == "自力自在":
        # 主導力と対応力を分割。自力単独より安定寄り、自在単独より主導力あり。
        jiryoku_comment_bonus = 0.065
        jizai_comment_bonus = 0.035
        if role == "head":
            jiryoku_comment_bonus += 0.010
        if role in ("head", "single"):
            jizai_comment_bonus += 0.005
        try:
            h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
            if h_line and int(h_line[0]) == int(no):
                jiryoku_comment_bonus += 0.015
        except Exception:
            pass
        if is_girls_like:
            jiryoku_comment_bonus *= 0.60
            jizai_comment_bonus *= 0.60

    elif move_style == "自在":
        # 自在は1着固定力ではなく、崩れにくさ・位置取りの安定として軽く加点する。
        jizai_comment_bonus = 0.065
        if role in ("head", "single"):
            jizai_comment_bonus += 0.010
        if is_girls_like:
            jizai_comment_bonus *= 0.60

    jiryoku_comment_bonus = clamp(jiryoku_comment_bonus, 0.0, 0.145)
    jizai_comment_bonus = clamp(jizai_comment_bonus, 0.0, 0.080)

    # -----------------------------------------------------
    # 単騎コメント補正
    #   ライン入力上の単騎とは別に、「一人で」「単騎で」「決めず」を明示する補助。
    #   強く減点せず、ライン保護・軸信頼の過信を少し抑える。
    # -----------------------------------------------------
    single_comment_bonus = 0.0
    if is_single_comment:
        single_comment_bonus = -0.010
        if role != "single":
            single_comment_bonus -= 0.010
        if move_style in ("自力", "自力自在", "自在"):
            single_comment_bonus *= 0.50
        if is_girls_like:
            single_comment_bonus *= 0.50
    single_comment_bonus = clamp(single_comment_bonus, -0.020, 0.0)

    # -----------------------------------------------------
    # ライン連動補正
    #   後ろの選手が「番手・目標」チェックありなら、
    #   その前のライン先頭を少し格上げする。
    #   例：42で2が「小原君」なら、4を少し救う。
    # -----------------------------------------------------
    line_cushion_bonus = 0.0

    try:
        gid = car_to_group.get(int(no), None)
        members = line_def.get(gid, []) if gid is not None else []

        # 自分がそのラインの先頭かどうか
        is_line_head = bool(members and int(members[0]) == int(no))

        if is_line_head:
            behind_members = [int(x) for x in members[1:]]

            has_target_behind = any(
                bool(target_comment_map.get(int(x), False))
                for x in behind_members
            )

            if has_target_behind:
                # 番手・後位が前を指名しているなら、先頭車を少し救う
                line_cushion_bonus = 0.040

                # H主導ラインの先頭なら、ライン成立度を少し上乗せ
                try:
                    h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                    if h_line and int(h_line[0]) == int(no):
                        line_cushion_bonus += 0.020
                except Exception:
                    pass

    except Exception:
        line_cushion_bonus = 0.0

    line_cushion_bonus = clamp(line_cushion_bonus, 0.0, 0.060)

    # -----------------------------------------------------
    # 競り補正
    #   ライン入力は崩さず、競り当事者を減点する。
    #   ・自分が競りチェックあり
    #   ・または他車から競り相手として指定されている
    #   このどちらかなら競り当事者として扱う。
    #   さらに、競り相手同士で基礎点が低い側は追加減点する。
    # -----------------------------------------------------
    seri_penalty = 0.0

    is_seri_involved = bool(is_seri_comment or seri_opponents)

    if is_seri_involved:
        seri_penalty = -0.100

        try:
            my_base = float(prof_base.get(int(no), 0.0))
            opp_bases = [
                float(prof_base.get(int(x), 0.0))
                for x in seri_opponents
                if int(x) in prof_base
            ]
            if opp_bases:
                best_opp = max(opp_bases)
                # 弱い側はより競り負け・脚消耗しやすいので追加減点
                if my_base + 1e-9 < best_opp:
                    seri_penalty -= 0.050
                else:
                    seri_penalty -= 0.020
        except Exception:
            pass

        # 番手で競る場合は、ライン連動が壊れやすい
        if role == "second":
            seri_penalty -= 0.030

        # ガールズは基本的に競りの意味が薄いので弱め
        if is_girls_like:
            seri_penalty *= 0.50

    seri_penalty = clamp(seri_penalty, -0.180, 0.0)

    # -----------------------------------------------------
    # 3番手以降の追走信頼補正
    # -----------------------------------------------------
    line_follow_trust_bonus = _line_follow_trust_bonus_for_car(no, role, is_girls_like)

    # =====================================================
    # 環境・個人補正（既存）
    # =====================================================
    wind     = _wind_func(eff_wind_dir, float(eff_wind_speed or 0.0), role, float(prof_escape[no]))
    bank_b   = bank_character_bonus(bank_angle, straight_length, prof_escape[no], prof_sashi[no], bank_length)
    length_b = bank_length_adjust(bank_length, prof_oikomi[no])
    indiv    = extra_bonus.get(no, 0.0)
    stab     = stability_score(no)  # 安定度
    h_bonus  = h_home_bonus(no, role, H_Z)

    l200 = l200_adjust(
        role, straight_length, bank_length, race_class,
        float(prof_escape[no]), float(prof_sashi[no]), float(prof_oikomi[no]),
        is_wet=st.session_state.get("is_wet", False)
    )

    # =====================================================
    # 合計スコア
    # =====================================================
    total_raw = (
        prof_base[no]
        + wind
        + cf["spread"] * level_rating_scale * tens_corr.get(no, 0.0)
        + bank_b
        + length_b
        + laps_adj
        + indiv
        + stab
        + h_bonus
        + l200
        + jiryoku_comment_bonus
        + jizai_comment_bonus
        + single_comment_bonus
        + line_cushion_bonus
        + seri_penalty
        + line_follow_trust_bonus
    )

    rows.append([
        no, role,
        round(prof_base[no], 3),
        round(wind, 3),
        round(cf["spread"] * level_rating_scale * tens_corr.get(no, 0.0), 3),
        round(bank_b, 3),
        round(length_b, 3),
        round(laps_adj, 3),
        round(indiv, 3),
        round(stab, 3),
        round(h_bonus, 3),
        round(l200, 3),
        round(jiryoku_comment_bonus, 3),
        round(jizai_comment_bonus, 3),
        round(single_comment_bonus, 3),
        round(line_cushion_bonus, 3),
        round(seri_penalty, 3),
        round(line_follow_trust_bonus, 3),
        float(total_raw)
    ])

df = pd.DataFrame(rows, columns=[
    "車番", "役割", "脚質基準(会場)", "風補正", "得点補正", "バンク補正",
    "周長補正", "周回補正", "個人補正", "安定度", "H補正", "ラスト200",
    "自力コメント補正", "自在コメント補正", "単騎コメント補正", "ライン連動補正", "競り補正", "後位信頼補正",
    "合計_SBなし_raw",
])

# ===== [PATCH] dfの型を確定させ、SBなし母集団(v_wo/v_final)を必ず作る =====
# 1) dfが空のときも落とさない
if df is None or len(df) == 0:
    st.warning("DEBUG: df（SBなし内訳）が空です。rowsが生成されていない可能性。")
    v_wo = {int(no): 0.0 for no in active_cars}
else:
    # 2) 車番を必ずintにする（★最重要：ここがズレると全部emptyになる）
    df["車番"] = df["車番"].astype(int)

    # 3) v_wo を df から必ず生成（全車キー保証）
    v_wo = {int(r["車番"]): float(r["合計_SBなし_raw"]) for _, r in df.iterrows()}
    for no in active_cars:
        ino = int(no)
        if ino not in v_wo:
            v_wo[ino] = 0.0

# 4) v_final は最低でも v_wo を引き継ぐ（KOが走らない/空でも落ちない）
v_final = dict(v_wo)

# 5) df_sorted_pure をここで確定（アンカー選定が安定）
df_sorted_pure = pd.DataFrame({
    "車番": sorted([int(k) for k in v_final.keys()]),
    "合計_SBなし": [float(v_final[int(k)]) for k in sorted([int(k) for k in v_final.keys()])]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)


    


# === ここは df = pd.DataFrame(...) の直後に貼るだけ ===

# --- fallback: note_sections が無い環境でも落ちないように ---
ns = globals().get("note_sections", None)
if not isinstance(ns, list):
    ns = []
    globals()["note_sections"] = ns
note_sections = ns


# ❶ バンク分類を“みなし直線/周長”から決定（33 / 400 / 500）
def _bank_str_from_lengths(bank_length: float) -> str:
    try:
        bl = float(bank_length)
    except:
        bl = 400.0
    if bl <= 340.0:   # 333系
        return "33"
    elif bl >= 480.0: # 500系
        return "500"
    return "400"

# ❷ 会場の“有利脚質”セット
def _favorable_styles(bank_str: str) -> set[str]:
    if bank_str == "33":   # 33＝先行系・ライン寄り
        return {"逃げ", "マーク"}
    if bank_str == "500":  # 500＝差し・マーク寄り
        return {"差し", "マーク"}
    return {"まくり", "差し"}  # 既定=400

# ❸ 役割の日本語化（lineの並びから）
def _role_jp(no: int, line_def: dict) -> str:
    r = role_in_line(no, line_def)
    return {"head":"先頭","second":"番手","thirdplus":"三番手","single":"単騎"}.get(r, "単騎")


# ❹ 入力の“逃/捲/差/マ”から、その選手の実脚質を決定（同点時はライン位置でブレない決め方）
def _dominant_style(no: int) -> str:
    vec = [("逃げ", k_esc.get(no,0)), ("まくり", k_mak.get(no,0)),
           ("差し", k_sashi.get(no,0)), ("マーク", k_mark.get(no,0))]
    m = max(v for _,v in vec)
    cand = [s for s,v in vec if v == m and m > 0]
    if cand:
        # タイブレーク：先頭>番手>三番手>単騎 を優先（先行気味→差し→マークの順）
        pr = {"先頭":3,"番手":2,"三番手":1,"単騎":0}
        role = role_in_line(no, line_def)
        role_pr = {"head":"先頭","second":"番手","thirdplus":"三番手","single":"単騎"}.get(role,"単騎")
        if "逃げ" in cand: return "逃げ"
        # 残りはライン位置で“差し”優先、その次に“マーク”
        if "差し" in cand and pr.get(role_pr,0) >= 2: return "差し"
        if "マーク" in cand: return "マーク"
        return cand[0]
    # 出走履歴ゼロなら位置で決める
    role = role_in_line(no, line_def)
    return {"head":"逃げ","second":"差し","thirdplus":"マーク","single":"まくり"}.get(role,"まくり")

# ❺ Rider 構造体（このファイル上部で既に宣言済みなら再定義不要）
from dataclasses import dataclass
@dataclass
class Rider:
    num: int; hensa: float; line_id: int; role: str; style: str

# ❻ 偏差値（Tスコア）を “合計_SBなし_raw” から作る（なければ Form で代用）
# ❻ 安定版：偏差値（Tスコア）を安全に作る
def _hensa_map_from_df(df: pd.DataFrame) -> dict[int,float]:
    col = "合計_SBなし_raw" if "合計_SBなし_raw" in df.columns else None

    # 生値ベクトルを取る（欠損があればフォールバックして補完）
    base = []
    for no in active_cars:
        try:
            v = float(df.loc[df["車番"]==no, col].values[0]) if col else float(form_T_map[no])
        except:
            v = float(form_T_map[no])  # fallback（=従来 Form 偏差値）
        base.append(v)

    base = np.array(base, dtype=float)

    # === 分散チェック：標準偏差が小さすぎる場合の暴走回避 ===
    sd = np.std(base)
    if sd < 1e-6:   # ← 安定化の本丸
        # 全員ほぼ同じ → 差が「無い」ので偏差値の差も付けない
        return {no: 50.0 for no in active_cars}

    # 通常の偏差値化
    T = 50 + 10 * (base - np.mean(base)) / sd

    # 浮動誤差対策で丸め
    T = np.clip(T, 20, 80)

    return {no: float(T[i]) for i,no in enumerate(active_cars)}


# ❼ RIDERS を“実データ”で構築（脚質は ❹、偏差値は ❻）
bank_str = _bank_str_from_lengths(bank_length)
hensa_map = _hensa_map_from_df(df)
RIDERS = []
for no in active_cars:
    # ラインIDは“そのラインの先頭車番”を代表IDに
    gid = None
    for g, mem in line_def.items():
        if no in mem:
            gid = mem[0]; break
    if gid is None: gid = no
    RIDERS.append(
        Rider(
            num=int(no),
            hensa=float(hensa_map[no]),
            line_id=int(gid),
            role=_role_jp(no, line_def),
            style=_dominant_style(no),
        )
    )

# ❽ フォーメーション（本命−2−全）：1列目=有利脚質内の偏差値最大
def _pick_axis(riders: list[Rider], bank_str: str) -> Rider:
    fav = _favorable_styles(bank_str)
    cand = [r for r in riders if r.style in fav]
    if not cand:
        raise ValueError(f"有利脚質{sorted(fav)}に該当0（bank={bank_str} / style誤りの可能性）")
    return max(cand, key=lambda r: r.hensa)

def _role_priority(bank_str: str) -> dict[str,int]:
    return ({"マーク":3,"番手":2,"三番手":1,"先頭":0} if bank_str=="33"
            else {"番手":3,"マーク":2,"三番手":1,"先頭":0})

from typing import Optional, List

def _pick_support(riders: List["Rider"], first: "Rider", bank_str: str) -> Optional["Rider"]:
    pr = _role_priority(bank_str)
    same = [r for r in riders if r.line_id==first.line_id and r.num!=first.num]
    if not same:
        return None
    same.sort(key=lambda r: (pr.get(r.role,0), r.hensa), reverse=True)
    return same[0]


# 印（◎→▲→偏差値補完）
def _read_marks_idmap() -> dict[int,str]:
    rm = globals().get("result_marks") or globals().get("marks") or {}
    out={}
    if isinstance(rm, dict):
        if any(isinstance(k,int) or (isinstance(k,str) and k.isdigit()) for k in rm.keys()):
            for k,v in rm.items():
                try: out[int(k)] = ("○" if str(v) in ("○","〇") else str(v))
                except: pass
        else:
            for sym,vid in rm.items():
                try: out[int(vid)] = ("○" if str(sym) in ("○","〇") else str(sym))
                except: pass
    return out

def _pick_partner(riders: list[Rider], used: set[int]) -> int|None:
    id2sym = _read_marks_idmap()
    for want in ("◎","▲"):
        t = next((i for i,s in id2sym.items() if i not in used and s==want), None)
        if t is not None: return t
    # 補完：偏差値上位
    rest = sorted([r for r in riders if r.num not in used], key=lambda r: r.hensa, reverse=True)
    return rest[0].num if rest else None

def make_trio_formation_final(riders: list[Rider], bank_str: str) -> str:
    first = _pick_axis(riders, bank_str)
    support = _pick_support(riders, first, bank_str)
    used = {first.num} | ({support.num} if support else set())
    partner = _pick_partner(riders, used)
    second = []
    if support: second.append(support.num)
    if partner is not None: second.append(partner)
    if len(second) < 2:
        # 2車に満たなければ偏差値補完
        rest = sorted([r.num for r in riders if r.num not in ({first.num}|set(second))],
                      key=lambda n: next(rr.hensa for rr in riders if rr.num==n),
                      reverse=True)
        if rest: second.append(rest[0])
    second = sorted(set(second))[:2]
    return f"三連複フォーメーション：{first.num}－{','.join(map(str, second))}－全"


mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0 * (df["合計_SBなし_raw"] - mu)

# --- SBなし(母集団) を df から「全車ぶん必ず」作る（None防止） ---
sb_map = {int(r["車番"]): float(r.get("合計_SBなし", 0.0)) for _, r in df.iterrows()}

# df が空 / sb_map が空のときは、全車0で母集団を作る（5車・欠番・SB未入力でも止めない）
if not sb_map:
    sb_map = {int(no): 0.0 for no in active_cars}
    

# === [PATCH-A] 安定度をENVから分離し、各柱をレース内z化（SD固定） ===
SD_FORM = 0.28
SD_ENV  = 0.20
SD_STAB = 0.12
SD_L200 = float(globals().get("SD_L200", 0.22))  # ← 追加。まず0.22〜0.30で様子見

# 安定度（raw）と、ENVのベース（= 合計_SBなし_raw から安定度だけ除いたもの）
STAB_RAW = {int(df.loc[i, "車番"]): float(df.loc[i, "安定度"]) for i in df.index}
ENV_BASE = {
    int(df.loc[i, "車番"]): (
        float(df.loc[i, "合計_SBなし_raw"])
        - float(df.loc[i, "安定度"])
        - float(df.loc[i, "ラスト200"])
    )
    for i in df.index
}

# ENV → z
_env_arr = np.array([float(ENV_BASE.get(n, np.nan)) for n in active_cars], dtype=float)
_mask = np.isfinite(_env_arr)
if int(_mask.sum()) >= 2:
    mu_env = float(np.mean(_env_arr[_mask])); sd_env = float(np.std(_env_arr[_mask]))
else:
    mu_env, sd_env = 0.0, 1.0
_den_env = (sd_env if sd_env > 1e-12 else 1.0)
ENV_Z = {int(n): (float(ENV_BASE.get(n, mu_env)) - mu_env) / _den_env for n in active_cars}

# FORM（すでに form_T_map は作ってある前提） → z
FORM_Z = {int(n): (float(form_T_map.get(n, 50.0)) - 50.0) / 10.0 for n in active_cars}

# STAB（安定度 raw） → z
_stab_arr = np.array([float(STAB_RAW.get(n, np.nan)) for n in active_cars], dtype=float)
_m2 = np.isfinite(_stab_arr)
if int(_m2.sum()) >= 2:
    mu_st = float(np.mean(_stab_arr[_m2])); sd_st = float(np.std(_stab_arr[_m2]))
else:
    mu_st, sd_st = 0.0, 1.0
_den_st = (sd_st if sd_st > 1e-12 else 1.0)
STAB_Z = {int(n): (float(STAB_RAW.get(n, mu_st)) - mu_st) / _den_st for n in active_cars}

# L200（残脚）→ z
_l200_arr = np.array([float(L200_RAW.get(n, np.nan)) for n in active_cars], dtype=float)
_m3 = np.isfinite(_l200_arr)
if int(_m3.sum()) >= 2:
    mu_l2 = float(np.mean(_l200_arr[_m3])); sd_l2 = float(np.std(_l200_arr[_m3]))
else:
    mu_l2, sd_l2 = 0.0, 1.0
_den_l2 = (sd_l2 if sd_l2 > 1e-12 else 1.0)
L200_Z = {int(n): (float(L200_RAW.get(n, mu_l2)) - mu_l2) / _den_l2 for n in active_cars}

# ===== KO方式（印に混ぜず：展開・ケンで利用） =====

# 0) SBなし(母集団) を df から確実に作る（全車）
sb_map = {int(k): float(v) for k, v in zip(df["車番"].astype(int), df["合計_SBなし"].astype(float))}

# ★必須：dfが空でも全車0で母集団を作る
if not sb_map:
    sb_map = {int(no): 0.0 for no in active_cars}

# 1) key 欠損チェック
missing = [int(n) for n in active_cars if int(n) not in sb_map]
if missing:
    st.error(f"SBなし(母集団) が欠損してる車番: {missing} / sb_map.keys={sorted(sb_map.keys())}")
    # st.stop()

# 2) 値が None/NaN チェック
bad = [
    int(n) for n in active_cars
    if (int(n) in sb_map) and (
        sb_map[int(n)] is None or
        (isinstance(sb_map[int(n)], float) and np.isnan(sb_map[int(n)]))
    )
]
if bad:
    st.error(f"SBなし(母集団) の値が None/NaN: {bad} / values={[sb_map[int(n)] for n in bad]}")
    # st.stop()

# 3) KO入力に使う母集団（全車）
v_wo = dict(sb_map)

# 4) 以降 KO
_is_girls = is_girls_like
head_scale = KO_HEADCOUNT_SCALE.get(int(n_cars), 1.0)
ko_scale_raw = (KO_GIRLS_SCALE if _is_girls else 1.0) * head_scale
KO_SCALE_MAX = 0.45
ko_scale = min(ko_scale_raw, KO_SCALE_MAX)

if ko_scale > 0.0 and line_def and len(line_def) >= 1 and v_wo:
    # --- KO順序（_ko_order が落ちる/不正でも必ずフォールバックで作る） ---
    try:
        ko_order = _ko_order(
            v_wo, line_def, S, B,
            line_factor=line_factor_eff,
            gap_delta=KO_GAP_DELTA
        )
    except Exception as e:
        # Streamlitで原因を見たいならコメント解除
        # st.warning(f"_ko_order fallback: {type(e).__name__}: {e}")
        ko_order = None

    # ★重要：ko_order が None/空/欠損でも「全車」を必ず含める
    ko_order = [int(c) for c in (ko_order or []) if int(c) in v_wo]
    rest = [int(c) for c in v_wo.keys() if int(c) not in set(ko_order)]
    rest = sorted(rest, key=lambda c: float(v_wo[int(c)]), reverse=True)
    ko_order = ko_order + rest  # ← 全車を必ず含める（ここが最重要）

    # ここ以降は ko_order が必ず全車になるので安全
    vals = [float(v_wo[c]) for c in v_wo.keys()]
    mu0  = float(np.mean(vals))
    sd0  = float(np.std(vals) + 1e-12)
    KO_STEP_SIGMA_LOCAL = max(0.25, KO_STEP_SIGMA * 0.7)
    step = KO_STEP_SIGMA_LOCAL * sd0
    # ★new_scores は「全車のベース」から開始して KO で上書き
    new_scores = dict(v_wo)

    for rank, car in enumerate(ko_order, start=1):
        rank_adjust = step * (len(ko_order) - rank)
        blended = (1.0 - ko_scale) * float(v_wo[int(car)]) + ko_scale * (
            mu0 + rank_adjust - (len(ko_order)/2.0 - 0.5)*step
        )
        new_scores[int(car)] = float(blended)

    v_final = dict(new_scores)

else:
    # KOしない時も「全車保持」
    if v_wo:
        ko_order = sorted(v_wo.keys(), key=lambda c: float(v_wo[c]), reverse=True)
        v_final = dict(v_wo)
    else:
        ko_order = []
        v_final = {}

# --- 純SBなしランキング（KOまで／格上げ前）
df_sorted_pure = (pd.DataFrame({
    "車番": sorted([int(k) for k in v_final.keys()]),
    "合計_SBなし": [round(float(v_final[int(c)]), 6) for c in sorted([int(k) for k in v_final.keys()])]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True))


# ===== 印用（既存の安全弁を維持） =====
FINISH_WEIGHT   = globals().get("FINISH_WEIGHT", 6.0)
FINISH_WEIGHT_G = globals().get("FINISH_WEIGHT_G", 3.0)
POS_BONUS  = globals().get("POS_BONUS", {0: 0.0, 1: -0.6, 2: -0.9, 3: -1.2, 4: -1.4})
POS_WEIGHT = globals().get("POS_WEIGHT", 1.0)
SMALL_Z_RATING = globals().get("SMALL_Z_RATING", 0.01)
FINISH_CLIP = globals().get("FINISH_CLIP", 4.0)
TIE_EPSILON  = globals().get("TIE_EPSILON", 0.8)

# --- p2のZ化など（従来どおり） ---
p2_list = [float(p2_eff.get(n, 0.0)) for n in active_cars]
if len(p2_list) >= 1:
    mu_p2  = float(np.mean(p2_list))
    sd_p2  = float(np.std(p2_list) + 1e-12)
else:
    mu_p2, sd_p2 = 0.0, 1.0
p2z_map = {n: (float(p2_eff.get(n, 0.0)) - mu_p2) / sd_p2 for n in active_cars}
p1_eff_safe = {n: float(p1_eff.get(n, 0.0)) if 'p1_eff' in globals() and p1_eff is not None else 0.0 for n in active_cars}
p2only_map = {n: max(0.0, float(p2_eff.get(n, 0.0)) - float(p1_eff_safe.get(n, 0.0))) for n in active_cars}
zt = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
zt_map = {n: float(zt[i]) for i, n in enumerate(active_cars)} if active_cars else {}


# === [PATCH-1] ENV/FORM をレース内で z 化し、目標SDを掛ける（anchor_score の前に置く） ===
SD_FORM = 0.28   # Balanced 既定
SD_ENV  = 0.20

# ENV = v_final（風・会場・周回疲労・個人補正・安定度 等を含む“Form以外”）
# ENV = v_final を int キー前提に揃える
_env_arr = np.array([float(v_final.get(int(n), np.nan)) for n in active_cars], dtype=float)

_mask = np.isfinite(_env_arr)
if int(_mask.sum()) >= 2:
    mu_env = float(np.mean(_env_arr[_mask]))
    sd_env = float(np.std(_env_arr[_mask]))
else:
    mu_env, sd_env = 0.0, 1.0

_den = sd_env if sd_env > 1e-12 else 1.0
ENV_Z = {int(n): (float(v_final.get(int(n), mu_env)) - mu_env) / _den for n in active_cars}


# FORM = form_T_map（T=50, SD=10）→ z 化
FORM_Z = {int(n): (float(form_T_map.get(n, 50.0)) - 50.0) / 10.0 for n in active_cars}


# --- ここで必ず定義してから使う（NameError防止） ---
line_sb_enable = bool(globals().get("line_sb_enable", (race_class != "ガールズ")))

def _pos_idx(no: int) -> int:
    g = car_to_group.get(no)
    if g is None or g not in line_def:
        return 4  # 単騎/不明は最後方（POS_BONUS[4]）

    grp = line_def[g]  # 例: [5,2,6] みたいな並び
    try:
        return max(0, grp.index(no))
    except ValueError:
        return 4  # グループに居ないなら最後方扱い


bonus_init, _ = compute_lineSB_bonus(
    line_def, S, B,
    line_factor=line_factor_eff,
    exclude=None, cap=cap_SB_eff,
    enable=line_sb_enable
)

def anchor_score(no: int) -> float:
    role = role_in_line(no, line_def)
    sb = float(
        bonus_init.get(car_to_group.get(no, None), 0.0)
        * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
    )
    pos_term = (POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)) if line_sb_enable else 0.0
    env_term  = SD_ENV  * float(ENV_Z.get(int(no), 0.0))
    form_term = SD_FORM * float(FORM_Z.get(int(no), 0.0))
    stab_term = SD_STAB * float(STAB_Z.get(int(no), 0.0))
    l200_term = SD_L200 * float(L200_Z.get(int(no), 0.0))
    tiny      = SMALL_Z_RATING * float(zt_map.get(int(no), 0.0))
    return env_term + form_term + stab_term + l200_term + sb + pos_term + tiny




# === デバッグ表示（必要なときだけ / anchor_score定義の後, 印出力の前） ===
# for no in active_cars:
#     role = role_in_line(no, line_def)
#     sb_dbg  = bonus_init.get(car_to_group.get(no, None), 0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0)
#     pos_dbg = POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)
#     form_dbg = SD_FORM * FORM_Z.get(no, 0.0)
#     env_dbg  = SD_ENV  * ENV_Z.get(no, 0.0)
#     stab_dbg = (SD_STAB * STAB_Z.get(no, 0.0)) if 'STAB_Z' in globals() else 0.0
#     tiny_dbg = SMALL_Z_RATING * zt_map.get(no, 0.0)

#     total = form_dbg + env_dbg + stab_dbg + sb_dbg + pos_dbg + tiny_dbg
#     st.write(no, {
#         "form": round(form_dbg, 4),
#         "env":  round(env_dbg, 4),
#         "stab": round(stab_dbg, 4),
#         "sb":   round(sb_dbg, 4),
#         "pos":  round(pos_dbg, 4),
#         "tiny": round(tiny_dbg, 4),
#         "TOTAL(anchor_score期待値)": round(total, 4),
#     })



# ===== ◎候補抽出（既存ロジック維持）
cand_sorted = sorted(active_cars, key=lambda n: anchor_score(n), reverse=True)
C = cand_sorted[:min(3, len(cand_sorted))]
ratings_sorted2 = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank2 = {n: i+1 for i,n in enumerate(ratings_sorted2)}
ALLOWED_MAX_RANK = globals().get("ALLOWED_MAX_RANK", 5)

guarantee_top_rating = True
if guarantee_top_rating and (race_class == "ガールズ") and len(ratings_sorted2) >= 1:
    top_rating_car = ratings_sorted2[0]
    if top_rating_car not in C:
        C = [top_rating_car] + [c for c in C if c != top_rating_car]
        C = C[:min(3, len(cand_sorted))]

ANCHOR_CAND_SB_TOPK   = globals().get("ANCHOR_CAND_SB_TOPK", 5)
ANCHOR_REQUIRE_TOP_SB = globals().get("ANCHOR_REQUIRE_TOP_SB", 3)

# ===== ANCHOR 選定（SBなし母集団ベース）+ 安全弁 + DEBUG =====
ANCHOR_CAND_SB_TOPK   = globals().get("ANCHOR_CAND_SB_TOPK", 5)
ANCHOR_REQUIRE_TOP_SB = globals().get("ANCHOR_REQUIRE_TOP_SB", 3)

# --- DEBUG（必要ならOFFにできる） ---
DBG_ANCHOR = bool(globals().get("DBG_ANCHOR", True))

def _safe_int(x, default=1):
    try:
        return int(x)
    except Exception:
        return int(default)

# df_sorted_pure が空なら、active_cars を母集団として使う（落下防止）
df_pure_empty = (df_sorted_pure is None) or (len(df_sorted_pure) == 0)

if df_pure_empty:
    base_order = [int(x) for x in list(active_cars)[:]]  # 1..7
else:
    # 念のため int 化
    base_order = df_sorted_pure["車番"].astype(int).tolist()

# rank_pure（SBなしランキング順位）
rank_pure = {int(no): i + 1 for i, no in enumerate(base_order)}

# 候補プール：C の中で SBなし上位K位
cand_pool = [int(c) for c in C if rank_pure.get(int(c), 999) <= ANCHOR_CAND_SB_TOPK]

# もし空なら、SBなし上位K位から直接作る
if not cand_pool:
    cand_pool = [int(no) for no in base_order[:min(ANCHOR_CAND_SB_TOPK, len(base_order))]]

# 最終フォールバック（どれも無い場合）
fallback_no = int(active_cars[0]) if active_cars else 1

# anchor_no_pre（まずは候補プール内で anchor_score 最大）
if cand_pool:
    anchor_no_pre = max(cand_pool, key=lambda x: anchor_score(int(x)))
else:
    anchor_no_pre = fallback_no

anchor_no = anchor_no_pre

# 同点圏（TIE_EPSILON以内）なら ratings_rank2 で決める
top2 = sorted(cand_pool, key=lambda x: anchor_score(int(x)), reverse=True)[:2]
if len(top2) >= 2:
    s1 = float(anchor_score(int(top2[0])))
    s2 = float(anchor_score(int(top2[1])))
    if (s1 - s2) < TIE_EPSILON:
        better_by_rating = min(top2, key=lambda x: ratings_rank2.get(int(x), 999))
        anchor_no = int(better_by_rating)

# SBなし上位N位縛り
if rank_pure.get(int(anchor_no), 999) > ANCHOR_REQUIRE_TOP_SB:
    pool = [int(c) for c in cand_pool if rank_pure.get(int(c), 999) <= ANCHOR_REQUIRE_TOP_SB]
    if pool:
        anchor_no = max(pool, key=lambda x: anchor_score(int(x)))
    else:
        anchor_no = int(base_order[0]) if base_order else fallback_no

    st.caption(
        f"※ ◎は『SBなし 上位{ANCHOR_REQUIRE_TOP_SB}位以内』縛りで {anchor_no_pre}→{anchor_no} に調整。"
    )



# ===== confidence 算出（anchor_score のギャップ/分散）=====
role_map = {int(no): role_in_line(int(no), line_def) for no in active_cars}

cand_scores = [float(anchor_score(int(no))) for no in C] if len(C) >= 2 else [0.0, 0.0]
cand_scores_sorted = sorted(cand_scores, reverse=True)
conf_gap = float(cand_scores_sorted[0] - cand_scores_sorted[1]) if len(cand_scores_sorted) >= 2 else 0.0

# v_final が空のときは spread=0 で落ちないように（confidenceは混戦寄りになる）
spread = float(np.std(list(v_final.values()))) if isinstance(v_final, dict) and len(v_final) >= 2 else 0.0
norm = conf_gap / (spread if spread > 1e-6 else 1.0)
confidence = "優位" if norm >= 1.0 else ("互角" if norm >= 0.5 else "混戦")

# ===== 格上げ（v_final が空でも落ちないように）=====
if not isinstance(v_final, dict) or len(v_final) == 0:
    # downstream を落とさないための最小母集団（全車0）
    v_final = {int(no): 0.0 for no in active_cars}

score_adj_map = apply_anchor_line_bonus(v_final, car_to_group, role_map, int(anchor_no), confidence)

df_sorted_wo = pd.DataFrame({
    "車番": [int(c) for c in active_cars],
    "合計_SBなし": [
        round(float(score_adj_map.get(int(c), v_final.get(int(c), float("-inf")))), 6)
        for c in active_cars
    ]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

velobi_wo = list(zip(
    df_sorted_wo["車番"].astype(int).tolist(),
    df_sorted_wo["合計_SBなし"].round(3).tolist()
))
# ==============================
# ★ レース内T偏差値 → 印 → 買い目 → note出力（2車系対応＋会場個性浸透版）
# ==============================
import math
import numpy as np
import pandas as pd
import streamlit as st

import re
from typing import List

def parse_line_str(line_str: str) -> List[List[int]]:
    s = (line_str or "").strip()
    if not s:
        return []
    s = s.replace("　", " ")
    groups = [g for g in s.split(" ") if g]
    lines = []
    for g in groups:
        nums = [int(ch) for ch in re.findall(r"\d", g)]
        if nums:
            lines.append(nums)
    return lines

def initial_queue_from_lines(lines: List[List[int]]) -> List[int]:
    q = []
    used = set()
    for group in lines:
        for n in group:
            if n not in used:
                q.append(n)
                used.add(n)
    return q

def estimate_finaljump_queue(initial_queue: List[int], score_rank: List[int], k: float = 2.2) -> List[int]:
    if not score_rank:
        return []
    if not initial_queue:
        return score_rank[:]
    pos = {n: i for i, n in enumerate(initial_queue)}
    nmax = max(len(score_rank), 1)
    power = {n: (nmax - i) for i, n in enumerate(score_rank)}  # 1位が最大
    def key(n: int) -> float:
        p0 = pos.get(n, 10_000)
        pw = power.get(n, 0)
        return p0 - k * pw
    return sorted(score_rank, key=key)

def arrow_format(order: List[int]) -> str:
    return " → ".join(str(n) for n in order)


HEN_DEC_PLACES = 1
EPS = 1e-12

# ====== ユーティリティ ======
def coerce_score_map(d, n_cars: int) -> dict[int, float]:
    out: dict[int, float] = {}
    t = str(type(d)).lower()
    if "pandas.core.frame" in t:
        df_ = d
        car_col = "車番" if "車番" in df_.columns else None
        if car_col is None:
            for c in df_.columns:
                if np.issubdtype(df_[c].dtype, np.integer):
                    car_col = c; break
        score_col = None
        for cand in ["合計_SBなし","SBなし","スコア","score","SB_wo","SB"]:
            if cand in df_.columns:
                score_col = cand; break
        if score_col is None:
            for c in df_.columns:
                if c == car_col: continue
                if np.issubdtype(df_[c].dtype, np.number):
                    score_col = c; break
        if car_col is not None and score_col is not None:
            for _, r in df_.iterrows():
                try:
                    i = int(r[car_col]); x = float(r[score_col])
                except Exception:
                    continue
                out[i] = x
    elif "pandas.core.series" in t:
        for k, v in d.to_dict().items():
            try:
                i = int(k); x = float(v)
            except Exception:
                continue
            out[i] = x
    elif hasattr(d, "items"):
        for k, v in d.items():
            try:
                i = int(k); x = float(v)
            except Exception:
                continue
            out[i] = x
    elif isinstance(d, (list, tuple, np.ndarray)):
        arr = list(d)
        if len(arr) == n_cars and all(not isinstance(x,(list,tuple,dict)) for x in arr):
            for idx, v in enumerate(arr, start=1):
                try: out[idx] = float(v)
                except Exception: out[idx] = np.nan
        else:
            for it in arr:
                if isinstance(it,(list,tuple)) and len(it) >= 2:
                    try:
                        i = int(it[0]); x = float(it[1])
                        out[i] = x
                    except Exception:
                        continue
    for i in range(1, int(n_cars)+1):
        out.setdefault(i, np.nan)
    return out









def _format_rank_from_array(ids, arr):
    pairs = [(i, float(arr[idx])) for idx, i in enumerate(ids)]
    pairs.sort(key=lambda kv: ((1,0) if not np.isfinite(kv[1]) else (0,-kv[1]), kv[0]))
    return " ".join(str(i) for i,_ in pairs)

# ====== ここから処理本体 ======

# 1) 母集団車番
try:
    USED_IDS = sorted(int(i) for i in (active_cars if active_cars else range(1, n_cars+1)))
except Exception:
    USED_IDS = list(range(1, int(n_cars)+1))
M = len(USED_IDS)

# 2) SBなしのソース（df優先→velobi_wo）
score_map_from_df = coerce_score_map(globals().get("df_sorted_wo", None), n_cars)
score_map_vwo     = coerce_score_map(globals().get("velobi_wo", None),   n_cars)
SB_BASE_MAP = score_map_from_df if any(np.isfinite(list(score_map_from_df.values()))) else score_map_vwo

# 偏差値母集団は「SBなし（KO適用後＆格上げ前後どちらか）」に固定
SB_BASE_MAP = {int(i): float(score_adj_map.get(int(i), v_final.get(int(i), np.nan))) for i in USED_IDS}



# 3) スコア配列（スコア順表示と偏差値母集団を共用）
xs_base_raw = np.array([SB_BASE_MAP.get(i, np.nan) for i in USED_IDS], dtype=float)

# 4) 偏差値T（レース内：平均50・SD10、NaN→50）
xs_race_t, mu_sb, sd_sb, k_finite = t_score_from_finite(xs_base_raw)




missing = ~np.isfinite(xs_base_raw)
if missing.any():
    sb_for_sort = {i: SB_BASE_MAP.get(i, -1e18) for i in USED_IDS}
    idxs = np.where(missing)[0].tolist()
    idxs.sort(key=lambda ii: (-float(sb_for_sort.get(USED_IDS[ii], -1e18)), USED_IDS[ii]))
    k = len(idxs); delta = 0.12; center = (k - 1)/2.0 if k > 1 else 0.0
    for r, ii in enumerate(idxs):
        xs_race_t[ii] = 50.0 + delta * (center - r)

# 5) dict化・表示用
race_t = {USED_IDS[idx]: float(round(xs_race_t[idx], HEN_DEC_PLACES)) for idx in range(M)}

# === 5.5) クラス別ライン偏差値ボーナス（ライン間→ライン内：低T優先 3:2:1） ===
# クラス別の総ポイント（Girlsは無効）
CLASS_LINE_POOL = {
    "Ｓ級":           21.0,
    "Ａ級":           15.0,
    "Ａ級チャレンジ":  9.0,
    "ガールズ":        0.0,
}
pool_total = float(CLASS_LINE_POOL.get(race_class, 0.0))

def _line_rank_weights(n_lines: int) -> list[float]:
    # 2本: 3:2 / 3本: 5:4:3 / 4本以上: 6,5,4,3,2,1...
    if n_lines <= 1: return [1.0]
    if n_lines == 2: return [3.0, 2.0]
    if n_lines == 3: return [5.0, 4.0, 3.0]
    base = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    if n_lines <= len(base): return base[:n_lines]
    ext = base[:]
    while len(ext) < n_lines:
        ext.append(max(1.0, ext[-1]-1.0))
    return ext[:n_lines]

def _in_line_weights(members_sorted_lowT_first: list[int]) -> dict[int, float]:
    # ライン内は「低T優先で 3:2:1、4人目以降0」→合計1に正規化
    raw = [3.0, 2.0, 1.0]
    w = {}
    for i, car in enumerate(members_sorted_lowT_first):
        w[int(car)] = (raw[i] if i < len(raw) else 0.0)
    s = sum(w.values())
    return {k: (v/s if s > 0 else 0.0) for k, v in w.items()}

_lines = list((globals().get("line_def") or {}).values())
if pool_total > 0.0 and _lines:
    # ライン強度＝そのラインの race_t 平均
    line_scores = []
    for mem in _lines:
        if not mem: 
            continue
        avg_t = float(np.mean([race_t.get(int(c), 50.0) for c in mem]))
        line_scores.append((tuple(mem), avg_t))
    # 強い順に並べてライン間ポイント配分
    line_scores.sort(key=lambda x: (-x[1], x[0]))
    rank_w = _line_rank_weights(len(line_scores))
    sum_rank_w = float(sum(rank_w)) if rank_w else 1.0
    line_share = {}
    for (mem, _avg), wr in zip(line_scores, rank_w):
        line_share[mem] = pool_total * (float(wr) / sum_rank_w)

    # 各ラインの配分を「低T→高T」の順に 3:2:1 で割り振り
    bonus_map = {int(i): 0.0 for i in USED_IDS}
    for mem, share in line_share.items():
        mem = list(mem)
        mem_sorted_lowT = sorted(mem, key=lambda c: (race_t.get(int(c), 50.0), int(c)))
        w_in = _in_line_weights(mem_sorted_lowT)  # 合計1
        for car in mem_sorted_lowT:
            bonus_map[int(car)] += share * w_in[int(car)]

    # 偏差値に加算（xs_race_tが計算本体。race_tは表示用に丸め直す）
    for idx, car in enumerate(USED_IDS):
        add = float(bonus_map.get(int(car), 0.0))
        xs_race_t[idx] = float(xs_race_t[idx]) + add
        race_t[int(car)] = float(round(xs_race_t[idx], HEN_DEC_PLACES))
# ← この後に既存の race_z 計算が続く



# ==============================
# 偏差値テーブル（SBなし母集団）＋欠損ガード
# ==============================
race_z = (xs_race_t - 50.0) / 10.0

# --- SBなし(母集団) を map として確定（KO入力もここを使う） ---
# USED_IDS と xs_base_raw は「同じ順番」で対応している前提
sb_map = {}
for cid, x in zip(USED_IDS, xs_base_raw):
    try:
        if x is None:
            continue
        xf = float(x)
        if not np.isfinite(xf):
            continue
        sb_map[int(cid)] = xf
    except Exception:
        pass

# --- 欠損チェック（None連発の犯人特定） ---
missing = [int(n) for n in active_cars if int(n) not in sb_map]
if missing:
    st.error(f"SBなし(母集団) が欠損してる車番: {missing} / sb_map.keys={sorted(sb_map.keys())}")


# zipで短くなってる可能性チェック
if len(xs_base_raw) != len(USED_IDS):
    st.error("xs_base_raw と USED_IDS の長さが一致していません。zip が途中で切れて欠損になります。")


# --- 表（hen_df）を sb_map から作る：Noneは明示的にNoneで残す ---
hen_df = pd.DataFrame({
    "車": USED_IDS,
    "SBなし(母集団)": [sb_map.get(int(cid), None) for cid in USED_IDS],
    "偏差値T(レース内)": [race_t[int(cid)] for cid in USED_IDS],
}).sort_values(["偏差値T(レース内)", "車"], ascending=[False, True]).reset_index(drop=True)

st.markdown("### 偏差値（レース内T＝平均50・SD10｜SBなしと同一母集団）")
st.caption(f"μ={mu_sb if np.isfinite(mu_sb) else 'nan'} / σ={sd_sb:.6f} / 有効件数k={k_finite}")
st.dataframe(hen_df, use_container_width=True)

# 7) 印（◎〇▲）＝ T↓ → SBなし↓ → 車番↑（βは除外）
if "select_beta" not in globals():
    def select_beta(cars): return None
if "enforce_alpha_eligibility" not in globals():
    def enforce_alpha_eligibility(m): return m

# ===== βラベル付与（単なる順位ラベル） =====
def assign_beta_label(result_marks: dict[str, int], used_ids: list[int], df_sorted) -> dict[str, int]:
    marks = dict(result_marks)
    # 6車以下は出さない（集計仕様）
    if len(used_ids) <= 6:
        return marks
    # 既にβがあれば何もしない
    if "β" in marks:
        return marks
    try:
        last_car = int(df_sorted.loc[len(df_sorted) - 1, "車番"])
        if last_car not in marks.values():
            marks["β"] = last_car
    except Exception:
        pass
    return marks


# ===== 印の採番（β廃止→無印で保持）========================================
# 依存: USED_IDS, race_t, xs_base_raw, line_def, car_to_group が上で定義済み

# スコアの補助（安定のため race_t 優先→同点は sb_base でタイブレーク）
sb_base = {
    int(USED_IDS[idx]): float(xs_base_raw[idx]) if np.isfinite(xs_base_raw[idx]) else float("-inf")
    for idx in range(len(USED_IDS))
}

def _race_t_val(i: int) -> float:
    try:
        return float(race_t.get(int(i), 50.0))
    except Exception:
        return 50.0

# === βは作らない。全員を候補にして上位から印を振る
seed_pool = list(map(int, USED_IDS))
order_by_T = sorted(
    seed_pool,
    key=lambda i: (-_race_t_val(i), -sb_base.get(i, float("-inf")), i)
)

result_marks: dict[str,int] = {}
reasons: dict[int,str] = {}

# ◎〇▲ を上位から
for mk, car in zip(["◎","〇","▲"], order_by_T):
    result_marks[mk] = int(car)

# ◎の同ラインを優先して残り印（△, ×, α）を埋める
line_def     = globals().get("line_def", {}) or {}
car_to_group = globals().get("car_to_group", {}) or {}
anchor_no    = result_marks.get("◎", None)

mates_sorted: list[int] = []
if anchor_no is not None:
    a_gid = car_to_group.get(anchor_no, None)
    if a_gid is not None and a_gid in line_def:
        used_now = set(result_marks.values())
        mates_sorted = sorted(
            [int(c) for c in line_def[a_gid] if int(c) not in used_now],
            key=lambda x: (-sb_base.get(int(x), float("-inf")), int(x))
        )

used = set(result_marks.values())
overall_rest = [int(c) for c in USED_IDS if int(c) not in used]
overall_rest = sorted(
    overall_rest,
    key=lambda x: (-sb_base.get(int(x), float("-inf")), int(x))
)

# 同ライン優先 → 残りスコア順
tail_priority = mates_sorted + [c for c in overall_rest if c not in mates_sorted]

for mk in ["△","×","α"]:
    if mk in result_marks:
        continue
    if not tail_priority:
        break
    no = int(tail_priority.pop(0))
    result_marks[mk] = no
    reasons[no] = f"{mk}（◎ライン優先→残りスコア順）"

# === 無印の集合（＝上の印が付かなかった残り全員）
marked_ids = set(result_marks.values())
no_mark_ids = [int(c) for c in USED_IDS if int(c) not in marked_ids]
# 表示はT優先・同点はsb_base
no_mark_ids = sorted(
    no_mark_ids,
    key=lambda x: (-_race_t_val(int(x)), -sb_base.get(int(x), float("-inf")), int(x))
)

# ===== 以降のUI出力での使い方 ==============================================
# ・印の一行（note用）: 既存の join を差し替え
#   例）(' '.join(f'{m}{result_marks[m]}' for m in ['◎','〇','▲','△','×','α'] if m in result_marks))
#   の直後などに「無」を追加
#   例）
#   ('無　' + (' '.join(map(str, no_mark_ids)) if no_mark_ids else '—'))
#
# ・以降のロジックでは「β」への参照を残さないこと（Noneチェック含め全削除OK）
#   もし `if i != result_marks.get("β")` のような行が残っていたら、単に削除してください。


if "α" not in result_marks:
    used_now = set(result_marks.values())
    pool = [i for i in USED_IDS if i not in used_now]
    if pool:
        alpha_pick = pool[-1]
        result_marks["α"] = alpha_pick
        reasons[alpha_pick] = reasons.get(alpha_pick, "α（フォールバック：禁止条件全滅→最弱を採用）")




# =========================
#  Tesla369｜出力統合・最終ブロック（安定版・重複なし / 3車ライン厚め対応）
# =========================
import re, json, hashlib, math
from typing import List, Dict, Any, Optional

# ---------- 基本ヘルパ ----------
def _t369_norm(s) -> str:
    return (str(s) if s is not None else "").replace("　", " ").strip()

def _t369_safe_mean(xs, default: float = 0.0) -> float:
    try:
        return sum(xs) / len(xs) if xs else default
    except Exception:
        return default

def _t369_sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-2.0 * x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# ---------- 文脈→ライン/印/スコア復元 ----------
def _t369_parse_lines_from_context() -> List[List[int]]:
    # _groups 優先
    try:
        _gs = globals().get("_groups") or []
        if _gs:
            out: List[List[int]] = []
            for g in _gs:
                ln = [int(x) for x in g if str(x).strip()]
                if ln: out.append(ln)
            if out: return out
    except Exception:
        pass
    # line_inputs（例："16","524","37"...）
    try:
        arr = [_t369_norm(x) for x in (globals().get("line_inputs") or []) if _t369_norm(x)]
        out: List[List[int]] = []
        for s in arr:
            nums = [int(ch) for ch in s if ch.isdigit()]
            if nums: out.append(nums)
        return out
    except Exception:
        return []

def _t369_lines_str(lines: List[List[int]]) -> str:
    return " ".join("".join(str(n) for n in ln) for ln in lines)

def _t369_buckets(lines: List[List[int]]) -> Dict[int, str]:
    m: Dict[int, str] = {}
    lid = 0
    for ln in lines:
        if len(ln) == 1:
            m[ln[0]] = f"S{ln[0]}"
        else:
            lid += 1
            for n in ln: m[n] = f"L{lid}"
    return m

# ライン
_lines_list: List[List[int]] = _t369_parse_lines_from_context()
lines_str: str = globals().get("lines_str") or _t369_lines_str(_lines_list)

# 印（result_marks → {"◎":3,...}）
_result_marks_raw = (globals().get("result_marks", {}) or {})
marks: Dict[str, int] = {}
for k, v in _result_marks_raw.items():
    m = re.search(r"\d+", str(v))
    if m:
        try: marks[str(k)] = int(m.group(0))
        except Exception: pass

# スコア（race_t / USED_IDS）
race_t   = dict(globals().get("race_t", {}) or {})
USED_IDS = list(globals().get("USED_IDS", []) or [])

def _t369_num(v) -> float:
    try: return float(v)
    except Exception:
        try: return float(str(v).replace("%","").strip())
        except Exception: return 0.0

def _t369_get_score_from_entry(e: Any) -> float:
    if isinstance(e, (int, float)): return float(e)
    if isinstance(e, dict):
        for k in ("偏差値","hensachi","dev","score","sc","S","s","val","value"):
            if k in e: return _t369_num(e[k])
    return 0.0

scores: Dict[int, float] = {}
ids_source = USED_IDS[:] or [n for ln in _lines_list for n in ln]
for n in ids_source:
    e = race_t.get(n, race_t.get(int(n), race_t.get(str(n), {})))
    scores[int(n)] = _t369_get_score_from_entry(e)
for n in [x for ln in _lines_list for x in ln]:
    scores.setdefault(int(n), 0.0)

# ---------- 流れ指標（簡潔・安定版） ----------
# ---------- 流れ指標（簡潔・安定版） ----------
def compute_flow_indicators(lines_str, marks, scores):
    parts = [_t369_norm(p) for p in str(lines_str).split() if _t369_norm(p)]
    lines = [[int(ch) for ch in p if ch.isdigit()] for p in parts if any(ch.isdigit() for ch in p)]
    if not lines:
        return {
            "VTX": 0.0, "FR": 0.0, "U": 0.0,
            "note": "【流れ未循環】ラインなし → ケン",
            "waves": {}, "vtx_bid": "", "lines": [], "dbg": {},
            "FR_line": [], "VTX_line": [], "U_line": []
        }

    buckets = _t369_buckets(lines)
    bucket_to_members = {buckets[ln[0]]: ln for ln in lines}

    def mean(xs, d=0.0):
        try:
            return sum(xs) / len(xs) if xs else d
        except Exception:
            return d

    def avg_score(mem):
        return mean([scores.get(n, 50.0) for n in mem], 50.0)

    muA = mean([avg_score(ln) for ln in lines], 50.0) / 100.0
    star_id = marks.get("◎", -999)
    none_id = marks.get("無", -999)

    def est(mem):
        A = max(10.0, min(avg_score(mem), 90.0)) / 100.0
        if star_id in mem:
            phi0, d = -0.8, +1
        elif none_id in mem:
            phi0, d = +0.8, -1
        else:
            phi0, d = +0.2, +1
        phi = phi0 + 1.2 * (A - muA)
        return A, phi, d

    def S_end(A, phi, t=0.9, f=0.9, gamma=0.12):
        return A * math.exp(-gamma * t) * (
            2 * math.pi * f * math.cos(2 * math.pi * f * t + phi)
            - gamma * math.sin(2 * math.pi * f * t + phi)
        )

    waves = {}
    for bid, mem in bucket_to_members.items():
        A, phi, d = est(mem)
        waves[bid] = {"A": A, "phi": phi, "d": d, "S": S_end(A, phi, t=0.9)}

    def I(bi, bj):
        if not bi or not bj or bi not in waves or bj not in waves:
            return 0.0
        return math.cos(waves[bi]["phi"] - waves[bj]["phi"])

    # ★順流/逆流：ライン強さ（スコア合計）で決める
    def line_strength(bid: str) -> float:
        mem = bucket_to_members.get(bid, [])
        return float(sum(scores.get(n, 50.0) for n in mem))

    all_buckets = list(bucket_to_members.keys())
    b_star = max(all_buckets, key=lambda bid: (line_strength(bid), bid))
    cand_buckets = [bid for bid in all_buckets if bid != b_star]
    b_none = min(cand_buckets, key=lambda bid: (line_strength(bid), bid)) if cand_buckets else ""

    # --- VTX ---
    vtx_list = []
    for bid, mem in bucket_to_members.items():
        if bid in (b_star, b_none):
            continue
        if waves.get(bid, {}).get("S", -1e9) < -0.02:
            continue
        wA = 0.5 + 0.5 * waves[bid]["A"]
        v = (0.6 * abs(I(bid, b_star)) + 0.4 * abs(I(bid, b_none))) * wA
        vtx_list.append((v, bid))
    vtx_list.sort(reverse=True, key=lambda x: x[0])
    VTX = vtx_list[0][0] if vtx_list else 0.0
    VTX_bid = vtx_list[0][1] if vtx_list else ""

    # --- FR ---
    ws, wn = waves.get(b_star, {}), waves.get(b_none, {})

    def S_point(w, t=0.95, f=0.9, gamma=0.12):
        if not w:
            return 0.0
        A, phi = w.get("A", 0.0), w.get("phi", 0.0)
        return A * math.exp(-gamma * t) * (
            2 * math.pi * f * math.cos(2 * math.pi * f * t + phi)
            - gamma * math.sin(2 * math.pi * f * t + phi)
        )

    blend_star = 0.6 * S_point(ws) + 0.4 * ws.get("S", 0.0)
    blend_none = 0.6 * S_point(wn) + 0.4 * wn.get("S", 0.0)

    def sig(x, k=3.0):
        try:
            return 1.0 / (1.0 + math.exp(-k * x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    sd_raw = (sig(-blend_star, 3.0) - 0.5) * 2.0
    nu_raw = (sig(blend_none, 3.0) - 0.5) * 2.0
    sd = max(0.0, sd_raw)
    nu = max(0.05, nu_raw)
    FR = sd * nu

    # --- U ---
    vtx_vals = [v for v, _ in vtx_list] or [0.0]
    vtx_mu = _t369_safe_mean(vtx_vals, 0.0)
    vtx_sd = (_t369_safe_mean([(x - vtx_mu) ** 2 for x in vtx_vals], 0.0)) ** 0.5
    vtx_hi = max(0.60, vtx_mu + 0.35 * vtx_sd)
    VTX_high = 1.0 if VTX >= vtx_hi else 0.0

    S_max = max(1e-6, max(abs(w["S"]) for w in waves.values()))
    S_noneN = max(0.0, wn.get("S", 0.0)) / S_max
    U_raw = sig(I(b_none, b_star), k=2.0)
    U = max(0.05, (0.6 * U_raw + 0.4 * S_noneN) * (1.0 if VTX_high > 0 else 0.8))

    def label(bid):
        mem = bucket_to_members.get(bid, [])
        return "".join(map(str, mem)) if mem else "—"

    note = "\n".join([
        f"【順流】◎ライン {label(b_star)}：失速危険 {'高' if FR >= 0.15 else ('中' if FR >= 0.05 else '低')}",
        f"【渦】候補ライン：{label(VTX_bid)}（VTX={VTX:.2f}）",
        f"【逆流】無ライン {label(b_none)}：U={U:.2f}（※判定基準内）",
    ])

    dbg = {"blend_star": blend_star, "blend_none": blend_none, "sd": sd, "nu": nu, "vtx_hi": vtx_hi}

    # ★パッチ2：内部で使ったラインを返す
    def members_of(bid: str) -> list[int]:
        return list(bucket_to_members.get(bid, []) or [])

    FR_line = members_of(b_star)
    VTX_line = members_of(VTX_bid)
    U_line = members_of(b_none)

    return {
        "VTX": VTX,
        "FR": FR,
        "U": U,
        "note": note,
        "waves": waves,
        "vtx_bid": VTX_bid,
        "lines": lines,
        "dbg": dbg,
        "FR_line": FR_line,
        "VTX_line": VTX_line,
        "U_line": U_line,
    }


# === v2.3: 相手4枠ロジック（3車厚め“強制保証”＋3番手保証(帯)＋U高域でも最大2枚まで許容）===

import re
from typing import List, Dict, Optional

def _t369p_parse_groups(lines_str: str) -> List[List[int]]:
    parts = re.findall(r'[0-9]+', str(lines_str or ""))
    groups: List[List[int]] = []
    for p in parts:
        g = [int(ch) for ch in p]
        if g:
            groups.append(g)
    return groups

def _t369p_find_line_of(num: int, groups: List[List[int]]) -> List[int]:
    for g in groups:
        if num in g:
            return g
    return []

def _t369p_line_avg(g: List[int], hens: Dict[int, float]) -> float:
    if not g:
        return -1e9
    return sum(hens.get(x, 0.0) for x in g) / len(g)

def _t369p_best_in_group(
    g: List[int],
    hens: Dict[int, float],
    exclude: Optional[int] = None
) -> Optional[int]:
    cand = [x for x in (g or []) if x != exclude]
    if not cand:
        return None
    return max(cand, key=lambda x: hens.get(x, 0.0), default=None)

def select_tri_opponents_v2(
    axis: int,
    lines_str: str,
    hens: Dict[int, float],              # 偏差値/スコアのマップ
    vtx: float,                          # 渦の強さ（0〜1）
    u: float,                            # 逆流の強さ（0〜1）
    marks: Dict[str, int],               # 印（{'◎':5, ...}）
    shissoku_label: str = "中",          # ◎ラインの「失速危険」：'低'/'中'/'高'
    vtx_line_str: Optional[str] = None,  # 渦候補ライン（例 '375'）
    u_line_str: Optional[str] = None,    # 逆流ライン（例 '63'）
    n_opps: int = 4,
    fr_v: float | None = None,           # レースFR（帯判定用）
) -> List[int]:

    # しきい値/ブースト
    U_HIGH       = 0.90
    THIRD_BOOST  = 0.18
    THICK_BASE   = 0.25
    AXIS_LINE_2P = 0.35

    # 3番手保証（FR帯）
    BAND_LO, BAND_HI = 0.25, 0.65
    THIRD_MIN = 40.0
    _FRv = float(fr_v or 0.0)

    groups     = _t369p_parse_groups(lines_str)
    axis_line  = _t369p_find_line_of(int(axis), groups)
    others_all = [x for g in groups for x in g if x != axis]

    vtx_group = _t369p_parse_groups(vtx_line_str)[0] if vtx_line_str else []
    u_group   = _t369p_parse_groups(u_line_str)[0]   if u_line_str   else []

    # FRライン（◎のライン。なければ平均最大ライン）
    g_star  = marks.get("◎")
    FR_line = _t369p_find_line_of(int(g_star), groups) if isinstance(g_star, int) else []
    if not FR_line and groups:
        FR_line = max(groups, key=lambda g: _t369p_line_avg(g, hens))

    thick_groups = [g for g in groups if len(g) >= 3]  # 3車(以上)ライン
    thick_others = [g for g in thick_groups if g != (axis_line or [])]
    best_thick_other = max(thick_others, key=lambda g: _t369p_line_avg(g, hens), default=None)

    # 必須候補
    picks_must: List[int] = []

    # ① 軸相方（番手）を強採用
    axis_partner = _t369p_best_in_group(axis_line, hens, exclude=axis) if axis_line else None
    if axis_partner is not None:
        picks_must.append(axis_partner)

    # ② 対抗ライン代表（平均偏差最大ラインの代表）
    other_lines = [g for g in groups if g != axis_line]
    best_other_line = max(other_lines, key=lambda g: _t369p_line_avg(g, hens), default=None)
    opp_rep = _t369p_best_in_group(best_other_line, hens, exclude=None) if best_other_line else None
    if opp_rep is not None:
        picks_must.append(opp_rep)

    # ③ 逆流代表（U高域のみ）。※3車u_groupは最大2枚まで許容
    u_rep = None
    if u >= U_HIGH:
        if u_group:
            u_rep = _t369p_best_in_group(u_group, hens, exclude=None)
        else:
            pool = [x for x in others_all if x not in (axis_line or [])]
            u_rep = max(pool, key=lambda x: hens.get(x, 0.0), default=None) if pool else None
        if u_rep is not None:
            picks_must.append(u_rep)

    # ④ スコアリング
    scores_local: Dict[int, float] = {x: 0.0 for x in others_all}
    for x in scores_local:
        scores_local[x] += hens.get(x, 0.0) / 100.0  # 土台

    # 軸ライン：相方強化＋同ライン控えめ
    if axis_partner is not None and axis_partner in scores_local:
        scores_local[axis_partner] += 1.50
    for x in (axis_line or []):
        if x not in (axis, axis_partner) and x in scores_local:
            scores_local[x] += 0.20

    # 対抗代表を加点
    if opp_rep is not None and opp_rep in scores_local:
        scores_local[opp_rep] += 1.20

    # U高域：代表強化＋“2枚目抑制（3車なら許容2まで）”
    if u >= U_HIGH and u_rep is not None and u_rep in scores_local:
        scores_local[u_rep] += 1.00
        if u_group:
            penalty = 0.15 if len(u_group) >= 3 else 0.40
            for x in u_group:
                if x != u_rep and x in scores_local:
                    scores_local[x] -= penalty

    # VTX境界の調律
    if vtx <= 0.55:
        if opp_rep is not None and opp_rep in scores_local:
            scores_local[opp_rep] += 0.40
        for x in (vtx_group or []):
            if x in scores_local:
                scores_local[x] -= 0.20
    elif vtx >= 0.60:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None) if vtx_group else None
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.50

    # ◎「失速=高」→ ◎本人を減点・番手を加点
    if isinstance(g_star, int) and shissoku_label == "高":
        g_line = _t369p_find_line_of(g_star, groups)
        g_ban  = _t369p_best_in_group(g_line, hens, exclude=g_star) if g_line else None
        if g_star in scores_local:
            scores_local[g_star] -= 0.60
        if g_ban is not None and g_ban in scores_local:
            scores_local[g_ban] += 0.70

    # ★ 3車(以上)ラインは厚め（基礎加点）
    for g3 in thick_groups:
        for x in g3:
            if x != axis and x in scores_local:
                scores_local[x] += THICK_BASE

    # 軸が3車(以上)なら同ライン2枚体制を厚め
    if axis_line and len(axis_line) >= 3:
        for x in axis_line:
            if x not in (axis, axis_partner) and x in scores_local:
                scores_local[x] += AXIS_LINE_2P

    # 渦/FRが3車(以上)なら中核を少し厚め
    if vtx_group and len(vtx_group) >= 3:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None)
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.30

    if FR_line and len(FR_line) >= 3:
        add_fr = 0.30 if shissoku_label != "高" else 0.15
        for x in FR_line:
            if x != axis and x in scores_local:
                scores_local[x] += add_fr

    # 3列目ブースト（“3番手”を軽く押す：ライン並びの3番手がいる前提）
    if axis_line and len(axis_line) >= 3:
        third = axis_line[2]
        if third in scores_local:
            scores_local[third] += THIRD_BOOST

    # まずは必須枠を採用（順序維持）
    def _unique_keep_order(xs: List[int]) -> List[int]:
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    picks = [x for x in _unique_keep_order(picks_must) if x in scores_local and x != axis]

    # 補充：スコア高い順。ただしU高域では u_group の人数上限（1 or 2）を守る
    def _same_group(a: int, b: int, group: List[int]) -> bool:
        return bool(group and a in group and b in group)

    for x, _sc in sorted(scores_local.items(), key=lambda kv: kv[1], reverse=True):
        if x in picks or x == axis:
            continue
        if u >= U_HIGH and u_group:
            limit = 2 if len(u_group) >= 3 else 1
            cnt_u = sum(1 for y in picks if y in u_group)
            if cnt_u >= limit and any(_same_group(x, y, u_group) for y in picks):
                continue
        picks.append(x)
        if len(picks) >= n_opps:
            break

    # ★ 強制保証１：軸が3車(以上)なら、相手4枠に同ライン2枚（相方＋もう1枚）を必ず確保
    if axis_line and len(axis_line) >= 3:
        axis_members = [x for x in axis_line if x != axis]
        present = [x for x in picks if x in axis_members]
        if len(present) < 2 and len(axis_members) >= 2:
            cand = max([x for x in axis_members if x not in picks], key=lambda x: hens.get(x, 0.0), default=None)
            if cand is not None:
                drop_cands = [x for x in picks if x not in axis_members and x != axis_partner]
                if drop_cands:
                    worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                    picks = [x for x in picks if x != worst] + [cand]

    # ★ 強制保証２：軸ライン以外で“最厚”の3車(以上)ラインは、相手4枠に最低2枚を確保
    if best_thick_other:
        have = [x for x in picks if x in best_thick_other]
        need = min(2, len(best_thick_other))
        while len(have) < need and len(picks) > 0:
            cand = max(
                [x for x in best_thick_other if x not in picks and x != axis],
                key=lambda x: hens.get(x, 0.0),
                default=None
            )
            if cand is None:
                break
            drop_cands = [x for x in picks if x not in best_thick_other and x != axis_partner]
            if not drop_cands:
                break
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            if worst == cand:
                break
            picks = [x for x in picks if x != worst] + [cand]
            have = [x for x in picks if x in best_thick_other]

    # 最終保険：不足分があれば偏差順で埋める
    if len(picks) < n_opps:
        rest = [x for x in others_all if x not in picks and x != axis]
        for x in sorted(rest, key=lambda x: hens.get(x, 0.0), reverse=True):
            picks.append(x)
            if len(picks) >= n_opps:
                break

    # ==== 3番手保証（FR帯 0.25〜0.65 限定）====
    if BAND_LO <= _FRv <= BAND_HI:
        target = axis_line if (axis_line and len(axis_line) >= 3) else (
            best_thick_other if (best_thick_other and len(best_thick_other) >= 3) else None
        )
        if target:
            g_sorted = sorted(target, key=lambda x: hens.get(x, 0.0), reverse=True)
            if len(g_sorted) >= 3:
                third = g_sorted[2]
                if (third not in picks) and (hens.get(third, 0.0) >= THIRD_MIN) and (third != axis):
                    drop_cands = [x for x in picks if (x not in target) and (x != axis_partner)]
                    if drop_cands:
                        worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                        if worst != third:
                            picks = [x for x in picks if x != worst] + [third]

    # --- 二車軸ロック（相方は絶対保持） ---
    if (axis_partner is not None) and (axis_partner not in picks):
        drop_cands = [x for x in picks if x != axis_partner]
        if drop_cands:
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            picks = [x for x in picks if x != worst] + [axis_partner]
        else:
            picks.append(axis_partner)

    # --- ユニーク＆サイズ調整（相方保護） ---
    seen = set()
    picks = [x for x in picks if not (x in seen or seen.add(x))]

    if len(picks) > n_opps:
        to_drop = len(picks) - n_opps
        cand = [x for x in picks if x != axis_partner]
        cand_sorted = sorted(cand, key=lambda x: scores_local.get(x, -1e9))
        for i in range(min(to_drop, len(cand_sorted))):
            if cand_sorted[i] in picks:
                picks.remove(cand_sorted[i])

    return picks

# === /v2.3 ===




def format_tri_1x4(axis: int, opps: List[int]) -> str:
    opps_sorted = ''.join(str(x) for x in sorted(opps))
    return f"{axis}-{opps_sorted}-{opps_sorted}"

# === PATCH（generate_tesla_bets の直前に挿入）==============================
# 前提：ファイル上部に import re があるならここでは不要（無ければ追加）
# 前提：typing を上で import 済みならここでは不要（無ければ追加）

# 軸選定用（generate_tesla_bets から呼ばれる）
def _topk(line, k, scores):
    line = list(line or [])
    return sorted(line, key=lambda x: (scores.get(x, -1.0), -int(x)), reverse=True)[:k]

def _t369p_parse_groups(lines_str: str):
    parts = re.findall(r"[0-9]+", str(lines_str or ""))
    groups = []
    for p in parts:
        g = [int(ch) for ch in p]
        if g:
            groups.append(g)
    return groups

def _t369p_find_line_of(num: int, groups):
    for g in groups:
        if num in g:
            return g
    return []

def _t369p_line_avg(g, hens):
    if not g:
        return -1e9
    return sum(hens.get(x, 0.0) for x in g) / len(g)

def _t369p_best_in_group(g, hens, exclude=None):
    cand = [x for x in (g or []) if x != exclude]
    if not cand:
        return None
    return max(cand, key=lambda x: hens.get(x, 0.0), default=None)


# ---- 相手4枠ロジック v2.3（3車厚め“強制保証”＋3番手保証(帯)＋U高域でも最大2枚許容）----
def select_tri_opponents_v2(
    axis: int,
    lines_str: str,
    hens: dict,              # {車番:int -> 偏差値/スコア:float}
    vtx: float,              # 渦の強さ（0〜1）
    u: float,                # 逆流の強さ（0〜1）
    marks: dict,             # {印:車番} or {車番:印} が来るので両対応
    shissoku_label: str = "中",
    vtx_line_str=None,
    u_line_str=None,
    n_opps: int = 4,
    fr_v: float | None = None,   # レースFR（帯判定用）
):
    # しきい値/ブースト
    U_HIGH       = 0.90
    THIRD_BOOST  = 0.18
    THICK_BASE   = 0.25
    AXIS_LINE_2P = 0.35

    # 3番手保証（FR帯）
    BAND_LO, BAND_HI = 0.25, 0.65
    THIRD_MIN = 40.0
    _FRv = float(fr_v or 0.0)

    groups     = _t369p_parse_groups(lines_str)
    axis_line  = _t369p_find_line_of(int(axis), groups)
    others_all = [x for g in groups for x in g if x != axis]

    vtx_group = _t369p_parse_groups(vtx_line_str)[0] if vtx_line_str else []
    u_group   = _t369p_parse_groups(u_line_str)[0]   if u_line_str   else []

    # --- ◎車番を marks から取得（{印:車番} / {車番:印} 両対応）---
    g_star = None
    if marks:
        # {印:車番} の可能性
        if all(isinstance(v, int) for v in marks.values()):
            g_star = marks.get("◎", None)
        else:
            # {車番:印} の可能性
            for cid, sym in marks.items():
                try:
                    if sym == "◎":
                        g_star = int(cid)
                        break
                except Exception:
                    pass

    # FRライン（◎のライン。なければ平均最大ライン）
    FR_line = _t369p_find_line_of(int(g_star), groups) if isinstance(g_star, int) else []
    if (not FR_line) and groups:
        FR_line = max(groups, key=lambda g: _t369p_line_avg(g, hens))

    # 3車(以上)ライン群と「軸以外の最厚」
    thick_groups     = [g for g in groups if len(g) >= 3]
    thick_others     = [g for g in thick_groups if g != (axis_line or [])]
    best_thick_other = max(thick_others, key=lambda g: _t369p_line_avg(g, hens), default=None)

    # --- 必須枠 ---
    picks_must = []

    # ① 軸相方（番手）
    axis_partner = _t369p_best_in_group(axis_line, hens, exclude=axis) if axis_line else None
    if axis_partner is not None:
        picks_must.append(axis_partner)

    # ② 対抗ライン代表（平均偏差最大ラインの代表）
    other_lines = [g for g in groups if g != axis_line]
    best_other_line = max(other_lines, key=lambda g: _t369p_line_avg(g, hens), default=None)
    opp_rep = _t369p_best_in_group(best_other_line, hens, exclude=None) if best_other_line else None
    if opp_rep is not None:
        picks_must.append(opp_rep)

    # ③ 逆流代表（U高域のみ）。※u_group が3車以上なら最大2枚許容
    u_rep = None
    if u >= U_HIGH:
        if u_group:
            u_rep = _t369p_best_in_group(u_group, hens, exclude=None)
        else:
            pool = [x for x in others_all if x not in (axis_line or [])]
            u_rep = max(pool, key=lambda x: hens.get(x, 0.0), default=None) if pool else None
        if u_rep is not None:
            picks_must.append(u_rep)

    # --- スコアリング ---
    scores_local = {x: 0.0 for x in others_all}
    for x in scores_local:
        scores_local[x] += hens.get(x, 0.0) / 100.0  # 土台

    # 軸ライン：相方強化、同ライン他は控えめ
    if axis_partner is not None and axis_partner in scores_local:
        scores_local[axis_partner] += 1.50
    for x in (axis_line or []):
        if x not in (axis, axis_partner) and x in scores_local:
            scores_local[x] += 0.20

    # 対抗代表の底上げ
    if opp_rep is not None and opp_rep in scores_local:
        scores_local[opp_rep] += 1.20

    # U高域：代表強化＋2枚目抑制（3車以上は緩め）
    if u >= U_HIGH and u_rep is not None and u_rep in scores_local:
        scores_local[u_rep] += 1.00
        if u_group:
            penalty = 0.15 if len(u_group) >= 3 else 0.40
            for x in u_group:
                if x != u_rep and x in scores_local:
                    scores_local[x] -= penalty

    # VTX境界の調律
    if vtx <= 0.55:
        if opp_rep is not None and opp_rep in scores_local:
            scores_local[opp_rep] += 0.40
        for x in (vtx_group or []):
            if x in scores_local:
                scores_local[x] -= 0.20
    elif vtx >= 0.60:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None) if vtx_group else None
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.50

    # ◎「失速=高」→ ◎本人を減点・番手を加点
    if isinstance(g_star, int) and shissoku_label == "高":
        g_line = _t369p_find_line_of(g_star, groups)
        g_ban  = _t369p_best_in_group(g_line, hens, exclude=g_star) if g_line else None
        if g_star in scores_local:
            scores_local[g_star] -= 0.60
        if g_ban is not None and g_ban in scores_local:
            scores_local[g_ban] += 0.70

    # ★ 3車(以上)ライン厚め：基礎加点＋“3列目”ブースト（各ラインの3番手）
    for g3 in thick_groups:
        for x in g3:
            if x != axis and x in scores_local:
                scores_local[x] += THICK_BASE
        g_sorted = sorted(g3, key=lambda x: hens.get(x, 0.0), reverse=True)
        if len(g_sorted) >= 3:
            third = g_sorted[2]
            if third != axis and third in scores_local:
                scores_local[third] += THIRD_BOOST

    # 軸が3車(以上)：同ライン2枚体制を強化
    if axis_line and len(axis_line) >= 3:
        for x in axis_line:
            if x not in (axis, axis_partner) and x in scores_local:
                scores_local[x] += AXIS_LINE_2P

    # 渦/FRが3車(以上)：中核を少し厚め
    if vtx_group and len(vtx_group) >= 3:
        best_vtx = _t369p_best_in_group(vtx_group, hens, exclude=None)
        if best_vtx is not None and best_vtx in scores_local:
            scores_local[best_vtx] += 0.30
    if FR_line and len(FR_line) >= 3:
        add_fr = 0.30 if shissoku_label != "高" else 0.15
        for x in FR_line:
            if x != axis and x in scores_local:
                scores_local[x] += add_fr

    # 必須（順序維持）
    def _unique_keep_order(xs):
        seen, out = set(), []
        for x in xs:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    picks = [x for x in _unique_keep_order(picks_must) if x in scores_local and x != axis]

    # 補充：スコア順。U高域では u_group の人数上限（1 or 2）を守る
    def _same_group(a, b, group):
        return bool(group and a in group and b in group)

    for x, _sc in sorted(scores_local.items(), key=lambda kv: kv[1], reverse=True):
        if x in picks or x == axis:
            continue
        if u >= U_HIGH and u_group:
            limit = 2 if len(u_group) >= 3 else 1
            cnt_u = sum(1 for y in picks if y in u_group)
            if cnt_u >= limit and any(_same_group(x, y, u_group) for y in picks):
                continue
        picks.append(x)
        if len(picks) >= n_opps:
            break

    # ★ 強制保証１：軸が3車(以上)→相手4枠に同ライン2枚（相方＋もう1枚）を確保
    if axis_line and len(axis_line) >= 3:
        axis_members = [x for x in axis_line if x != axis]
        present = [x for x in picks if x in axis_members]
        if len(present) < 2 and len(axis_members) >= 2:
            cand = max([x for x in axis_members if x not in picks], key=lambda x: hens.get(x, 0.0), default=None)
            if cand is not None:
                drop_cands = [x for x in picks if x not in axis_members and x != axis_partner]
                if drop_cands:
                    worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                    picks = [x for x in picks if x != worst] + [cand]

    # ★ 強制保証２：軸以外で“最厚”の3車(以上)ライン→相手4枠に最低2枚を確保
    if best_thick_other:
        have = [x for x in picks if x in best_thick_other]
        need = min(2, len(best_thick_other))
        while len(have) < need and len(picks) > 0:
            cand = max([x for x in best_thick_other if x not in picks and x != axis],
                       key=lambda x: hens.get(x, 0.0), default=None)
            if cand is None:
                break
            drop_cands = [x for x in picks if x not in best_thick_other and x != axis_partner]
            if not drop_cands:
                break
            worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
            if worst == cand:
                break
            picks = [x for x in picks if x != worst] + [cand]
            have = [x for x in picks if x in best_thick_other]

    # 最終保険：不足分を偏差順で埋める
    if len(picks) < n_opps:
        rest = [x for x in others_all if x not in picks and x != axis]
        for x in sorted(rest, key=lambda x: hens.get(x, 0.0), reverse=True):
            picks.append(x)
            if len(picks) >= n_opps:
                break

    # ===== 3番手保証（FR帯 0.25〜0.65）=====
    if (BAND_LO <= _FRv <= BAND_HI) and axis_line and len(axis_line) >= 3:
        g_sorted = sorted(axis_line, key=lambda x: hens.get(x, 0.0), reverse=True)
        if len(g_sorted) >= 3:
            axis_third = g_sorted[2]
            if (axis_third not in picks) and (hens.get(axis_third, 0.0) >= THIRD_MIN) and (axis_third != axis):
                drop_cands = [x for x in picks if (x not in axis_line) and (x != axis_partner)]
                if drop_cands:
                    worst = min(drop_cands, key=lambda x: scores_local.get(x, -1e9))
                    picks = [x for x in picks if x != worst] + [axis_third]

    # --- ユニーク＆サイズ調整（相方を落とさない） ---
    seen = set()
    uniq = []
    for x in picks:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    picks = uniq

    if len(picks) > n_opps:
        # 相方は保護して、残りから低スコアを落とす
        protect = set([axis_partner]) if axis_partner is not None else set()
        drop_pool = [x for x in picks if x not in protect]
        drop_pool_sorted = sorted(drop_pool, key=lambda x: scores_local.get(x, -1e9))
        while len(picks) > n_opps and drop_pool_sorted:
            picks.remove(drop_pool_sorted.pop(0))

    return picks


def _format_tri_axis_partner_rest(axis: int, opps: list, axis_line: list,
                                  hens: dict, lines: list) -> str:
    """
    出力形式： 軸・相方 － 残り3枠 － 残り3枠
    並び規則：対抗ラインの2名（番号昇順）→ 軸ラインの3番手（存在時）→ 残りをスコア順で充填
    """
    if not isinstance(axis, int) or axis <= 0 or not isinstance(opps, list):
        return "—"

    hens = {int(k): float(v) for k, v in (hens or {}).items() if str(k).isdigit()}
    axis_line = list(axis_line or [])

    # 相方（軸ライン内の最上位・軸以外）
    partner = None
    if axis in axis_line:
        cands = [x for x in axis_line if x != axis]
        if cands:
            partner = max(cands, key=lambda x: (hens.get(x, 0.0), -int(x)))

    # フォールバック：相方不在なら通常 1-XXXX-XXXX
    if partner is None:
        rest = "".join(str(x) for x in sorted(opps))
        return f"{axis}-{rest}-{rest}"

    # 軸3番手（スコア順の3番手）
    axis_third = None
    if len(axis_line) >= 3:
        g_sorted = sorted(axis_line, key=lambda x: hens.get(x, 0.0), reverse=True)
        if len(g_sorted) >= 3:
            axis_third = g_sorted[2]

    # 対抗ライン（＝軸ライン以外で平均偏差最大）
    def _line_avg(g):
        return sum(hens.get(x, 0.0) for x in g) / len(g) if g else -1e9
    other_lines = [g for g in (lines or []) if g != axis_line]
    opp_line = max(other_lines, key=_line_avg) if other_lines else []

    # 残り3枠（相方を除く）
    pool = [x for x in opps if x != partner]

    # まず対抗ラインの2名（昇順で最大2名）
    opp_two = sorted([x for x in pool if x in (opp_line or [])])[:2]

    rest_three = []
    rest_three.extend(opp_two)

    # 軸3番手を追加（まだ入っておらず、poolに居るなら）
    if axis_third is not None and axis_third in pool and axis_third not in rest_three:
        rest_three.append(axis_third)

    # 不足充填：スコア降順→番号昇順で埋める
    if len(rest_three) < 3:
        remain = [x for x in pool if x not in rest_three]
        remain_sorted = sorted(remain, key=lambda x: (hens.get(x, 0.0), -int(x)), reverse=True)
        rest_three.extend(remain_sorted[: (3 - len(rest_three))])

    rest_three = rest_three[:3]

    # 表示は「対抗(昇順) → それ以外」の順
    in_opp = [x for x in rest_three if x in (opp_line or [])]
    not_opp = [x for x in rest_three if x not in (opp_line or [])]
    rest_str = "".join(str(x) for x in (sorted(in_opp) + not_opp))

    return f"{axis}・{partner} － {rest_str} － {rest_str}"

# === /PATCH ==============================================================


# ======================= T369｜FREE-ONLY 完全置換ブロック（精簡版） =======================

# ---- 小ヘルパ（ローカル名で衝突回避） -----------------------------------------
def _free_fmt_nums(arr):
    if isinstance(arr, list):
        return "".join(str(x) for x in arr) if arr else "—"
    return "—"

def _free_norm_marks(marks_any):
    marks_any = dict(marks_any or {})
    if not marks_any:
        return {}
    # 値が全部 int → {印:車番} と判断し反転
    if all(isinstance(v, int) for v in marks_any.values()):
        out = {}
        for k, v in marks_any.items():
            try:
                out[int(v)] = str(k)
            except Exception:
                pass
        return out
    # それ以外は {車番:印}
    out = {}
    for k, v in marks_any.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            pass
    return out

def _free_fmt_marks_line(raw_marks: dict, used_ids: list) -> tuple[str, str]:
    """
    raw_marks: {車番:int -> '◎'} または { '◎' -> 車番:int } の両方に対応
    used_ids:  表示対象の車番リスト（スコア順など）
    戻り値: ("◎5 〇3 ▲1 △2 ×6 α7", "を除く未指名：...") のタプル
    """
    used_ids = [int(x) for x in (used_ids or [])]
    marks = _free_norm_marks(raw_marks)
    prio = ["◎", "〇", "▲", "△", "×", "α"]
    parts = []
    for s in prio:
        ids = [cid for cid, sym in marks.items() if sym == s]
        ids_sorted = sorted(ids, key=lambda c: (used_ids.index(c) if c in used_ids else 10**9, c))
        parts.extend([f"{s}{cid}" for cid in ids_sorted])
    marks_str = " ".join(parts)
    un = [cid for cid in used_ids if cid not in marks]
    no_str = ("を除く未指名：" + " ".join(map(str, un))) if un else ""
    return marks_str, no_str

# --- 3区分バンド（短評で使うなら残す） ---
def _band3_fr(fr: float) -> str:
    if fr >= 0.61: return "不利域"
    if fr >= 0.46: return "標準域"
    return "有利域"

def _band3_vtx(v: float) -> str:
    if v > 0.60:  return "不利域"
    if v >= 0.52: return "標準域"
    return "有利域"

def _band3_u(u: float) -> str:
    if u > 0.65:  return "不利域"
    if u >= 0.55: return "標準域"
    return "有利域"

# --- 優位/互角/混戦 判定（必要なら残す） ---
def infer_eval_with_share(fr_v: float, vtx_v: float, u_v: float, share_pct: float | None) -> str:
    fr_low, fr_high = 0.40, 0.60
    vtx_strong, u_strong = 0.60, 0.65
    share_lo, share_hi = 25.0, 33.0  # %
    if (fr_v > fr_high) and (vtx_v <= vtx_strong) and (u_v <= u_strong) and (share_pct is not None and share_pct >= share_hi):
        return "優位"
    if (fr_v < fr_low) or ((vtx_v > vtx_strong) and (u_v > u_strong)) or (share_pct is not None and share_pct <= share_lo):
        return "混戦"
    return "互角"

# ============================================================
# /T369｜FREE-ONLY 出力一括ブロック（券種コード完全撤去 + 0.000連発対策 + KO統一）
# ============================================================

def _normalize_lines(_lines):
    """
    入力 lines を必ず [[2,4],[5,7,1]...] の形にする
    - "24" / 24 / [24] / [2,4] どれでもOK（数字だけ抜いて桁分解）
    """
    out = []
    for ln in (_lines or []):
        if ln is None:
            continue
        s = "".join(ch for ch in str(ln) if ch.isdigit())
        if not s:
            continue
        out.append([int(ch) for ch in s])
    return out

# --- line_fr_map が無い/空でも出せる保険（本体は既存 _build_line_fr_map を優先） ---
if "_build_line_fr_map" not in globals():
    def _build_line_fr_map(lines, scores_map, FRv,
                           SINGLETON_FR_SCALE=0.70,
                           MIN_LINE_SHARE=0.00,
                           MAX_SINGLETON_SHARE=0.45):
        lines = _normalize_lines(lines)
        scores_map = {int(k): float(v) for k, v in (scores_map or {}).items() if str(k).strip().isdigit()}
        FRv = float(FRv or 0.0)
        if not lines:
            return {}

        line_sums = []
        for ln in lines:
            s = sum(scores_map.get(int(x), 0.0) for x in ln)
            if len(ln) == 1:
                s *= float(SINGLETON_FR_SCALE)
            line_sums.append((ln, s))

        total = sum(s for _, s in line_sums)
        sum_target = FRv if FRv > 0.0 else 1.0

        if total <= 0.0:
            eq = 1.0 / len(lines)
            return {"".join(map(str, ln)): eq for ln, _ in line_sums}

        return {"".join(map(str, ln)): sum_target * (s / total) for ln, s in line_sums}

# ---------- 3) 安全ラッパ（券種なし：flowだけ） ----------
def _safe_flow(lines_str, marks, scores):
    try:
        fr = compute_flow_indicators(lines_str, marks, scores)
        return fr if isinstance(fr, dict) else {}
    except Exception:
        return {}

# ===================== 4) 出力本体（券種なし・一括置換） =====================
try:
    import math

    # --- note_sections を必ず用意 ---
    ns = globals().get("note_sections", None)
    if not isinstance(ns, list):
        ns = []
        globals()["note_sections"] = ns
    note_sections = ns

    # ---- flow 作成 ----
    _flow = _safe_flow(
        globals().get("lines_str", ""),
        globals().get("marks", {}),
        globals().get("scores", {}),
    )
    globals()["_flow"] = _flow  # 後段参照用に保持

    # ---- 値の確定 ----
    FRv = float(_flow.get("FR", 0.0) or 0.0)
    VTXv = float(_flow.get("VTX", 0.0) or 0.0)
    Uv = float(_flow.get("U", 0.0) or 0.0)

    all_lines = list(_flow.get("lines") or [])
    all_lines = _normalize_lines(all_lines)  # ここで必ず正規化
    globals()["all_lines"] = all_lines

    # ---- レース名 ----
    venue = str(globals().get("track") or globals().get("place") or "").strip()
    race_no = str(globals().get("race_no") or "").strip()
    if venue or race_no:
        _rn = race_no if (race_no.endswith("R") or race_no == "") else f"{race_no}R"
        note_sections.append(f"{venue}{_rn}")
        note_sections.append("")

    # =========================================================
    # KO母集団スコア（v_final > v_wo > scores）で統一
    # =========================================================
    def _as_int_float_map(m):
        out = {}
        if not isinstance(m, dict):
            return out
        for k, v in m.items():
            try:
                kk = int(k)
                vv = float(v)
                if math.isfinite(vv):
                    out[kk] = vv
            except Exception:
                pass
        return out

    v_final_map = _as_int_float_map(globals().get("v_final"))
    v_wo_map = _as_int_float_map(globals().get("v_wo"))
    scores_map = _as_int_float_map(globals().get("scores"))

    score_map = dict(v_final_map or v_wo_map or scores_map or {})

    # active_cars を必ず含める（欠けを防ぐ）
    active_cars = [int(x) for x in (globals().get("active_cars") or []) if str(x).isdigit()]
    for n in active_cars:
        score_map.setdefault(int(n), 0.0)

   

        # =========================================================
    # KO母集団スコア補正：ライン3番手以降・H0/B0の過大評価抑制
    # ※脚質名に依存しない版。「追」ではなく「マーク」扱いでも効く。
    # =========================================================
    try:
        _line_def = globals().get("line_def", {})
        _H = globals().get("H", {})
        _B = globals().get("B", {})

        for _n in list(score_map.keys()):
            _car = int(_n)

            _role = role_in_line(_car, _line_def) if isinstance(_line_def, dict) else "single"

            _h_val = float(_H.get(_car, _H.get(str(_car), 0)) or 0)
            _b_val = float(_B.get(_car, _B.get(str(_car), 0)) or 0)

            # 例：364 の 4番 = thirdplus、H0、B0 → 必ず減点
            if _role == "thirdplus":
                if _h_val == 0 and _b_val == 0:
                    score_map[_n] = float(score_map[_n]) - 0.15
                else:
                    score_map[_n] = float(score_map[_n]) - 0.08

    except Exception as _e:
        note_sections.append(f"※KO母集団補正エラー：{_e}")

    score_map_before_last_half = dict(score_map)
    globals()["score_map_before_last_half"] = dict(score_map_before_last_half)

    # =========================================================
    # ラスト半周補正：自力粘り・番手差し
    # ※既存のKO母集団スコアに後付けする
    # =========================================================
    try:
        _line_def = globals().get("line_def", {})
        _H = globals().get("H", {})
        _B = globals().get("B", {})
        _kaku = globals().get("kaku", {})
        _tenscore = globals().get("tenscore", globals().get("tenscores", {}))

        # 競走得点の取り出し
        def _get_num_from_map(_mp, _car, _default=0.0):
            try:
                if isinstance(_mp, dict):
                    return float(_mp.get(int(_car), _mp.get(str(_car), _default)) or _default)
            except Exception:
                pass
            return float(_default)

        _race_scores = []
        for _n in active_cars:
            _v = _get_num_from_map(_tenscore, _n, 0.0)
            if _v > 0:
                _race_scores.append(_v)

        _race_avg_tenscore = float(np.mean(_race_scores)) if _race_scores else 0.0
        _last_half_bonus_map = {}
        _last_half_reason_map = {}
        
                # -------------------------------------------------
        # ラスト半周補正用：レース内順位マップ
        # 上位1/3判定用。7車なら3位以内。
        # -------------------------------------------------
        _active_list = [int(x) for x in active_cars]
        _top_third_limit = int(math.ceil(len(_active_list) / 3.0)) if _active_list else 3
        _top_third_limit = max(1, _top_third_limit)

        # 競走得点順位
        _race_score_rank_map = {}
        _ten_pairs = []
        for _n in _active_list:
            _v = _get_num_from_map(_tenscore, _n, 0.0)
            _ten_pairs.append((int(_n), float(_v)))

        _ten_pairs_sorted = sorted(_ten_pairs, key=lambda x: (-x[1], x[0]))
        for _idx, (_car2, _v2) in enumerate(_ten_pairs_sorted, start=1):
            _race_score_rank_map[int(_car2)] = int(_idx)

        # KO順位・展開順位
        # この時点の score_map_before_last_half は「ラスト半周補正前」のスコア
        _ko_score_rank_map = {}
        _ko_pairs_sorted = sorted(
            [(int(k), float(v)) for k, v in score_map_before_last_half.items()],
            key=lambda x: (-x[1], x[0])
        )
        for _idx, (_car2, _v2) in enumerate(_ko_pairs_sorted, start=1):
            _ko_score_rank_map[int(_car2)] = int(_idx)

        _tenkai_score_rank_map = dict(_ko_score_rank_map)

        # 順流・渦・逆流の複数上位は次段階用
        _scenario_top_count_map = globals().get("scenario_top_count_map", {})
        if not isinstance(_scenario_top_count_map, dict):
            _scenario_top_count_map = {}

        for _n in list(score_map.keys()):
            _car = int(_n)

            # ライン内の役割
            _role = role_in_line(_car, _line_def) if isinstance(_line_def, dict) else "single"

            # 同ライン先頭の競走得点
            _leader = _car
            try:
                if isinstance(_line_def, dict):
                    for _gid, _mem in _line_def.items():
                        _mem2 = [int(x) for x in _mem]
                        if _car in _mem2 and _mem2:
                            _leader = int(_mem2[0])
                            break
            except Exception:
                _leader = _car

            _car_ten = _get_num_from_map(_tenscore, _car, 0.0)
            _leader_ten = _get_num_from_map(_tenscore, _leader, _car_ten)

            _h_val = _get_num_from_map(_H, _car, 0.0)
            _b_val = _get_num_from_map(_B, _car, 0.0)

            # kakuは現在の入力仕様では使わない。
            # 関数互換用に空文字で渡す。
            _style = ""

            # H主導ラインの3番手以降かどうか
            _is_h_lead_thirdplus = False
            try:
                _h_members = []
                if home_top_gid is not None and isinstance(_line_def, dict):
                    _h_members = [int(x) for x in _line_def.get(home_top_gid, [])]

                if (
                    len(_h_members) >= 3
                    and _role == "thirdplus"
                    and _car in _h_members[2:]
                ):
                    _is_h_lead_thirdplus = True

            except Exception:
                _is_h_lead_thirdplus = False

            # ---------------------------------------------
            # ラスト半周用：個人成績率
            # x1 / x2 / x3 / x_out から
            # 1着率・2着内率・3着内率を作る
            # ---------------------------------------------
            _p1_rate = None
            _p2_rate = None
            _p3_rate = None

            try:
                _x1 = globals().get("x1", {})
                _x2 = globals().get("x2", {})
                _x3 = globals().get("x3", {})
                _xo = globals().get("x_out", {})

                _n1 = float(_x1.get(_car, _x1.get(str(_car), 0)) or 0)
                _n2 = float(_x2.get(_car, _x2.get(str(_car), 0)) or 0)
                _n3 = float(_x3.get(_car, _x3.get(str(_car), 0)) or 0)
                _no = float(_xo.get(_car, _xo.get(str(_car), 0)) or 0)

                _total = _n1 + _n2 + _n3 + _no

                if _total > 0:
                    _p1_rate = _n1 / _total
                    _p2_rate = (_n1 + _n2) / _total
                    _p3_rate = (_n1 + _n2 + _n3) / _total

            except Exception:
                _p1_rate = None
                _p2_rate = None
                _p3_rate = None

            _bonus, _reasons = calc_last_half_role_bonus(
                role=_role,
                kaku=_style,
                tenscore=_car_ten,
                leader_tenscore=_leader_ten,
                race_avg_tenscore=_race_avg_tenscore,
                h_count=_h_val,
                b_count=_b_val,
                race_score_rank=_race_score_rank_map.get(_car),
                ko_score_rank=_ko_score_rank_map.get(_car),
                tenkai_score_rank=_tenkai_score_rank_map.get(_car),
                top_third_limit=_top_third_limit,
                scenario_top_count=int(_scenario_top_count_map.get(_car, 0) or 0),
                p1_rate=_p1_rate,
                p2_rate=_p2_rate,
                p3_rate=_p3_rate,
            )

            _last_half_bonus_map[_car] = float(_bonus)
            _last_half_reason_map[_car] = list(_reasons)

            score_map[_car] = float(score_map.get(_car, 0.0)) + float(_bonus)

    


            # -------------------------------------------------
        # H主導ライン3番手以降：3着内率40%以上なら最低4番手評価まで床上げ
        # -------------------------------------------------
        THIRDPLUS_TOP3_RATE_MIN = 0.40
        THIRDPLUS_FLOOR_RANK = 4
        THIRDPLUS_FLOOR_EPS = 0.001

        def _normalize_rate_0to1(v):
            try:
                x = float(v)
                if x > 1.0:
                    x = x / 100.0
                return x
            except Exception:
                return None

        def _get_top3_rate_for_car(_car_no):
            """
            車番ごとの3着内率を取得する。
            変数名が多少違っても拾えるように、候補名とglobals内のdictを探す。
            値は 0.40 / 40.0 のどちらでも対応。
            """
            _car_no = int(_car_no)

            # よくありそうな名前を優先
            _candidate_names = [
                "top3_rate_map",
                "in3_rate_map",
                "pTop3_map",
                "ptop3_map",
                "car_top3_rate_map",
                "car_in3_rate_map",
                "top3_map",
                "in3_map",
                "P_TOP3_MAP",
                "IN3_RATE_MAP",
            ]

            for _name in _candidate_names:
                _obj = globals().get(_name, None)
                if isinstance(_obj, dict):
                    _v = _obj.get(_car_no, _obj.get(str(_car_no), None))
                    _r = _normalize_rate_0to1(_v)
                    if _r is not None:
                        return _r

            # 名前が違う場合の保険：globals内の「top3 / in3 / 3着」系dictを探索
            try:
                for _name, _obj in globals().items():
                    _lname = str(_name).lower()
                    if not isinstance(_obj, dict):
                        continue

                    if not (
                        "top3" in _lname
                        or "in3" in _lname
                        or "p_top3" in _lname
                        or "3着" in str(_name)
                        or "三着" in str(_name)
                    ):
                        continue

                    _v = _obj.get(_car_no, _obj.get(str(_car_no), None))
                    _r = _normalize_rate_0to1(_v)
                    if _r is not None:
                        return _r
            except Exception:
                pass

            return None



        globals()["last_half_bonus_map"] = _last_half_bonus_map
        globals()["last_half_reason_map"] = _last_half_reason_map
        globals()["score_map_last_half_applied"] = dict(score_map)

    except Exception as _e:
        note_sections.append(f"※ラスト半周補正エラー：{_e}")

    # =========================================================
    # 会場成績 × 最終ホームライン補正（買い目用スコア）
    # H1番手ラインはイン減速で減点、H2番手ラインは外スピードで加点
    # =========================================================
    try:
        _line_def = globals().get("line_def", {})
        _car_to_group = globals().get("car_to_group", {})
        _track = globals().get("track", st.session_state.get("track", ""))
        _venue_profile = globals().get("venue_profile", st.session_state.get("venue_profile", "unknown"))
        _home_top_gid = globals().get("home_top_gid", None)
        _home_second_gid = globals().get("home_second_gid", None)

        _home_flow_bonus_map = {}
        _home_flow_reason_map = {}
        _before_home_flow_map = dict(score_map)

        for _n in list(score_map.keys()):
            _car = int(_n)
            _role = role_in_line(_car, _line_def) if isinstance(_line_def, dict) else "single"
            _gid = _car_to_group.get(_car, None) if isinstance(_car_to_group, dict) else None

            _hf_bonus, _hf_reason = home_flow_adjust_by_venue(
                no=_car,
                role=_role,
                gid=_gid,
                home_top_gid=_home_top_gid,
                home_second_gid=_home_second_gid,
                track_name=_track,
                venue_profile=_venue_profile,
            )

            _home_flow_bonus_map[_car] = float(_hf_bonus)
            _home_flow_reason_map[_car] = str(_hf_reason)

            score_map[_car] = float(score_map.get(_car, 0.0)) + float(_hf_bonus)

        globals()["home_flow_bonus_map"] = dict(_home_flow_bonus_map)
        globals()["home_flow_reason_map"] = dict(_home_flow_reason_map)
        globals()["score_map_before_home_flow"] = dict(_before_home_flow_map)
        globals()["score_map_home_flow_applied"] = dict(score_map)

    except Exception as _e:
        note_sections.append(f"※会場×最終H補正エラー：{_e}")

    # =========================================================
    # v178：開催場決まり手補正（常時適用・雨天補正とは別枠）
    # 入力された1着/2着決まり手率を、役割別の小幅ptへ変換して加算。
    # =========================================================
    try:
        _vk_stats = globals().get("VENUE_KIMARITE_STATS", st.session_state.get("VENUE_KIMARITE_STATS", {}))
        _line_def_for_vk = globals().get("line_def", {})
        _before_vk_score_map = dict(score_map)

        score_map, _vk_role_bonus_map, _vk_reliability, _vk_detail, _vk_reason_map = _apply_venue_kimarite_to_score_map(
            score_map=score_map,
            line_def=_line_def_for_vk,
            stats=_vk_stats,
        )

        globals()["score_map_before_venue_kimarite"] = dict(_before_vk_score_map)
        globals()["score_map_venue_kimarite_applied"] = dict(score_map)
        globals()["venue_kimarite_role_bonus_map"] = dict(_vk_role_bonus_map)
        globals()["venue_kimarite_reliability"] = float(_vk_reliability)
        globals()["venue_kimarite_detail"] = dict(_vk_detail or {})
        globals()["venue_kimarite_reason_map"] = dict(_vk_reason_map)

    except Exception as _e:
        note_sections.append(f"※開催場決まり手補正エラー：{_e}")

    # 0/None/NaN の床値補完
    vals_pos = [
        float(v) for v in score_map.values()
        if isinstance(v, (int, float)) and float(v) > 0.0 and math.isfinite(float(v))
    ]

    _floor = min(vals_pos) if vals_pos else 1e-6

    for k in list(score_map.keys()):
        try:
            v = float(score_map[k])
            if (not math.isfinite(v)) or v <= 0.0:
                score_map[k] = float(_floor)
        except Exception:
            score_map[k] = float(_floor)

    globals()["score_map"] = score_map  # 後段参照用に保持

    # =========================================================
    # line_fr_map を確定（_lfr 未定義事故対策）
    # =========================================================
    line_fr_map = globals().get("line_fr_map")
    need_rebuild = (not isinstance(line_fr_map, dict)) or (len(line_fr_map) == 0)

    # 既存があればキー正規化（tuple/listキー → "571"）
    if (not need_rebuild) and isinstance(line_fr_map, dict):
        _lfm2 = {}
        for k, v in line_fr_map.items():
            try:
                if isinstance(k, (list, tuple, set)):
                    kk = "".join(str(x) for x in k if str(x).isdigit())
                else:
                    kk = "".join(ch for ch in str(k) if ch.isdigit())

                if kk:
                    _lfm2[kk] = float(v or 0.0)
            except Exception:
                continue

        line_fr_map = _lfm2
        need_rebuild = (len(line_fr_map) == 0)

    # 空なら作り直し
    if need_rebuild:
        try:
            line_fr_map = _build_line_fr_map(
                all_lines,
                score_map,
                FRv if FRv > 0.0 else 1.0
            )
        except Exception:
            line_fr_map = {}

    globals()["line_fr_map"] = line_fr_map

    def _line_key(ln):
        try:
            if not ln:
                return ""
            return "".join(str(int(x)) for x in ln if str(x).isdigit())
        except Exception:
            return "".join(ch for ch in str(ln) if ch.isdigit())

    def _lfr(ln):
        try:
            return float(line_fr_map.get(_line_key(ln), 0.0) or 0.0)
        except Exception:
            return 0.0
    # =========================================================
    # 展開評価（share_pct は「順流ライン」基準）
    # =========================================================
    FR_line = _flow.get("FR_line") or []
    VTX_line = _flow.get("VTX_line") or []
    U_line = _flow.get("U_line") or []

    FR_line = _normalize_lines([FR_line])[0] if FR_line else []
    VTX_line = _normalize_lines([VTX_line])[0] if VTX_line else []
    U_line = _normalize_lines([U_line])[0] if U_line else []

    globals()["FR_line"] = FR_line
    globals()["VTX_line"] = VTX_line
    globals()["U_line"] = U_line

    # =========================================================
    # 渦ラインを必ず埋める（空なら自動選定）
    # ルール：FR_line / U_line 以外で、想定FRが最大のラインを渦にする
    # =========================================================
    if (not VTX_line) or (_lfr(VTX_line) <= 0.0):
        _cand = []
        for ln in (all_lines or []):
            if not ln:
                continue
            if ln == FR_line or ln == U_line:
                continue
            _cand.append(ln)
        if _cand:
            VTX_line = max(_cand, key=lambda x: _lfr(x))
            globals()["VTX_line"] = VTX_line

    axis_line = FR_line if FR_line else (all_lines[0] if all_lines else [])
    axis_line_fr = float(line_fr_map.get(_line_key(axis_line), 0.0) or 0.0)
    total_fr = sum(float(v or 0.0) for v in line_fr_map.values()) if isinstance(line_fr_map, dict) else 0.0
    share_pct = (axis_line_fr / total_fr * 100.0) if (total_fr > 1e-12 and axis_line) else None

    note_sections.append(f"展開評価：{infer_eval_with_share(FRv, VTXv, Uv, share_pct)}")
    note_sections.append("")

    # ---- 時刻・クラス ----
    race_time = str(globals().get("race_time", "") or "")
    race_class = str(globals().get("race_class", "") or "")
    hdr = f"{race_time}　{race_class}".strip()
    if hdr:
        note_sections.append(hdr)

        # ---- ライン表示 ----
    line_inputs = globals().get("line_inputs", [])
    if isinstance(line_inputs, list) and any(str(x).strip() for x in line_inputs):
        _lines = [str(x).strip() for x in line_inputs if str(x).strip()]
        note_sections.append("ライン　" + "　".join(_lines))

        # H：最終ホーム想定ライン
        try:
            note_sections.append(f"最終ホーム想定　{home_line_text}")
            note_sections.append(f"H主導ライン　{home_top_line}")
        except Exception:
            pass

    note_sections.append("")

    # =========================================================
    # ライン想定FR（順流/渦/逆流 + その他）表示  ※区分け復活
    # =========================================================
    def _fmt_line(ln):
        try:
            f = globals().get("_free_fmt_nums")
            if callable(f):
                return f(ln)
        except Exception:
            pass
        return "".join(map(str, ln)) if isinstance(ln, (list, tuple)) and ln else "—"

        # =========================================================
    # ライン評価グループ（順流域／渦域／逆流域）
    # =========================================================
    def _fmt_line(ln):
        try:
            f = globals().get("_free_fmt_nums")
            if callable(f):
                return f(ln)
        except Exception:
            pass
        return "".join(map(str, ln)) if isinstance(ln, (list, tuple)) and ln else "—"

    def _same_line(a, b):
        return tuple(int(x) for x in (a or [])) == tuple(int(x) for x in (b or []))

    try:
        h_line_members = line_def.get(home_top_gid, []) if home_top_gid is not None else []
    except Exception:
        h_line_members = []

    valid_lines = [ln for ln in (all_lines or []) if ln]
    line_items = []

    for ln in valid_lines:
        fr = float(_lfr(ln))
        line_items.append({
            "line": ln,
            "fr": fr,
            "is_fr": _same_line(ln, FR_line),
            "is_vtx": _same_line(ln, VTX_line),
            "is_u": _same_line(ln, U_line),
            "is_h": _same_line(ln, h_line_members),
        })

    line_items = sorted(line_items, key=lambda x: (-x["fr"], _fmt_line(x["line"])))

    if line_items:
        top_fr = float(line_items[0]["fr"])

        # FR差による範囲判定
        # 7車以下はやや狭め、8・9車は広め
        if int(n_cars) >= 8:
            upper_gap = 0.080
            middle_ratio = 0.45
            h_gap = 0.150
        else:
            upper_gap = 0.050
            middle_ratio = 0.45
            h_gap = 0.090

        zones = {
            "順流域": [],
            "渦域": [],
            "逆流域": [],
        }

        for item in line_items:
            ln = item["line"]
            fr = float(item["fr"])
            gap = top_fr - fr
            ratio = (fr / top_fr) if top_fr > 1e-12 else 0.0

            tags = []
            if item["is_fr"]:
                tags.append("◎")
            if item["is_h"]:
                tags.append("H主導")
            if item["is_vtx"]:
                tags.append("旧渦")
            if item["is_u"]:
                tags.append("旧逆流")

            # 順流域：
            # FRトップ、またはFRトップとの差が小さいライン
            if item["is_fr"] or gap <= upper_gap:
                zone = "順流域"

            # H主導ラインは、FR2位級なら実質上位へ寄せる
            elif item["is_h"] and (gap <= h_gap or ratio >= 0.55):
                zone = "順流域"
                tags.append("実質上位")

            # 中位以上の別線は渦域
            elif ratio >= middle_ratio:
                zone = "渦域"

            # 低FR・単騎・押上げ側は逆流域
            else:
                zone = "逆流域"

            sort_score = fr + (0.030 if item["is_h"] else 0.0)

            zones[zone].append({
                "line": ln,
                "fr": fr,
                "tags": tags,
                "sort_score": sort_score,
            })

        for z in zones:
            zones[z] = sorted(
                zones[z],
                key=lambda x: (-x["sort_score"], -x["fr"], _fmt_line(x["line"]))
            )

        # =====================================================
        # v164: 順流域は必ず代表1ラインだけにする
        # 目的：157 と 24 のように複数ラインが同じ順流域へ入り、
        #       KO隊列で 15724 を1塊のように混ぜてしまう現象を防ぐ。
        #       単騎も1ラインとして扱う。
        # =====================================================
        try:
            jun_items = list(zones.get("順流域", []))
            if len(jun_items) > 1:
                # ◎ラインを最優先。なければ現在のソート順トップを順流代表にする。
                fr_items = [x for x in jun_items if "◎" in x.get("tags", [])]
                keep_jun = fr_items[0] if fr_items else jun_items[0]
                overflow = [x for x in jun_items if x is not keep_jun]

                zones["順流域"] = [keep_jun]

                # 余った順流候補は、まず渦域へ1本、残りは逆流域へ回す。
                # 既に渦域がある場合は、渦域へ追加してFR順で再ソートする。
                if overflow:
                    zones.setdefault("渦域", [])
                    zones.setdefault("逆流域", [])

                    if not zones.get("渦域"):
                        zones["渦域"].append(overflow[0])
                        zones["逆流域"].extend(overflow[1:])
                    else:
                        zones["渦域"].extend(overflow)

                    for _z in ("渦域", "逆流域"):
                        zones[_z] = sorted(
                            zones.get(_z, []),
                            key=lambda x: (-x["sort_score"], -x["fr"], _fmt_line(x["line"]))
                        )
        except Exception:
            pass

                # =====================================================
        # 全ラインが順流域に吸収された場合の強制分割
        # 目的：順流・渦・逆流メインが全部同じになるのを防ぐ
        # =====================================================
        try:
            if (
                len(zones.get("順流域", [])) >= 3
                and len(zones.get("渦域", [])) == 0
                and len(zones.get("逆流域", [])) == 0
            ):
                all_top_items = list(zones["順流域"])

                # まずFR順で並べる
                all_top_items = sorted(
                    all_top_items,
                    key=lambda x: (-float(x["fr"]), _fmt_line(x["line"]))
                )

                # ◎ラインは順流域に残す
                fr_items = [x for x in all_top_items if "◎" in x.get("tags", [])]

                if fr_items:
                    keep_jun = fr_items[0]
                else:
                    keep_jun = all_top_items[0]

                rest = [x for x in all_top_items if x is not keep_jun]

                # 残りの中でFR最上位を渦域へ
                rest = sorted(
                    rest,
                    key=lambda x: (-float(x["fr"]), _fmt_line(x["line"]))
                )

                keep_vtx = rest[0] if rest else None
                rest2 = [x for x in rest if x is not keep_vtx]

                zones["順流域"] = [keep_jun]
                zones["渦域"] = [keep_vtx] if keep_vtx is not None else []
                zones["逆流域"] = rest2

        except Exception:
            pass

        


        # =====================================================
        # v235: 順流・渦・逆流は必ず3枠に割り振る
        # 目的：ライン評価グループで逆流域が空なのに、流れ比率だけ逆流100%になる矛盾を防ぐ。
        # ・旧逆流タグを持つラインは逆流域の補完候補として最優先
        # ・旧渦タグを持つラインは渦域の補完候補として最優先
        # ・3ライン以上ある場合、表示上も内部比率上も3枠を空にしない
        # =====================================================
        try:
            _zone_names = ["順流域", "渦域", "逆流域"]
            for _z in _zone_names:
                zones.setdefault(_z, [])

            def _move_one_zone(_from, _to, _prefer_tag=None):
                try:
                    _items = list(zones.get(_from, []) or [])
                    if len(_items) <= 1:
                        return False
                    _idx = None
                    if _prefer_tag:
                        for _i, _it in enumerate(_items):
                            if _prefer_tag in (_it.get("tags", []) or []):
                                _idx = _i
                                break
                    if _idx is None:
                        # FRが低いものほど逆流/補完側へ回しやすい。
                        _idx = min(range(len(_items)), key=lambda i: (float(_items[i].get("fr", 0.0) or 0.0), _fmt_line(_items[i].get("line"))))
                    _item = _items.pop(_idx)
                    zones[_from] = _items
                    zones.setdefault(_to, [])
                    zones[_to].append(_item)
                    zones[_to] = sorted(
                        zones.get(_to, []),
                        key=lambda x: (-x.get("sort_score", 0.0), -float(x.get("fr", 0.0) or 0.0), _fmt_line(x.get("line")))
                    )
                    return True
                except Exception:
                    return False

            _all_zone_count = sum(len(zones.get(_z, []) or []) for _z in _zone_names)
            if _all_zone_count >= 3:
                # 逆流域が空なら、旧逆流タグを持つ渦域ラインを最優先で逆流域へ戻す。
                if not zones.get("逆流域"):
                    if not _move_one_zone("渦域", "逆流域", "旧逆流"):
                        _move_one_zone("順流域", "逆流域", "旧逆流")

                # 渦域が空なら、旧渦タグを持つ逆流域ラインを最優先で渦域へ戻す。
                if not zones.get("渦域"):
                    if not _move_one_zone("逆流域", "渦域", "旧渦"):
                        _move_one_zone("順流域", "渦域", "旧渦")

                # 順流域が空になる異常時だけ、最大FRのラインを順流域へ補完する。
                if not zones.get("順流域"):
                    _donors = [z for z in ("渦域", "逆流域") if len(zones.get(z, []) or []) > 1]
                    if _donors:
                        _from = max(_donors, key=lambda z: max(float(x.get("fr", 0.0) or 0.0) for x in zones.get(z, []) or []))
                        _items = list(zones.get(_from, []) or [])
                        _idx = max(range(len(_items)), key=lambda i: float(_items[i].get("fr", 0.0) or 0.0))
                        _item = _items.pop(_idx)
                        zones[_from] = _items
                        zones["順流域"] = [_item]

            # 3枠確定後のFR比率を保存。以後の流れ想定比率はこの表示分類を優先する。
            _zone_fr = {
                "順流": sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("順流域", []) or [])),
                "渦":   sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("渦域", []) or [])),
                "逆流": sum(float(x.get("fr", 0.0) or 0.0) for x in (zones.get("逆流域", []) or [])),
            }
            _zone_total = sum(_zone_fr.values())
            if _zone_total > 0:
                globals()["FLOW_RATIO_MAP_BY_ZONE"] = {
                    "順流": _zone_fr["順流"] / _zone_total,
                    "逆流": _zone_fr["逆流"] / _zone_total,
                    "渦": _zone_fr["渦"] / _zone_total,
                }
        except Exception:
            pass

        # KO隊列用：ラインごとの新ゾーン分類を保存
        _LINE_ZONE_MAP = {}

        _zone_to_short = {
            "順流域": "順流",
            "渦域": "渦",
            "逆流域": "逆流",
        }

        for zone_name, items in zones.items():
            short_zone = _zone_to_short.get(zone_name, "その他")
            for item in items:
                try:
                   key = "".join(ch for ch in str(item["line"]) if ch.isdigit())
                   if key:
                       _LINE_ZONE_MAP[key] = short_zone
                except Exception:
                   pass

        globals()["LINE_ZONE_MAP"] = _LINE_ZONE_MAP

        # st.write("DEBUG LINE_ZONE_MAP", _LINE_ZONE_MAP)

        note_sections.append("【ライン評価グループ】")

        for zone_name in ["順流域", "渦域", "逆流域"]:
            items = zones.get(zone_name, [])
            if not items:
                note_sections.append(f"{zone_name}：—")
                continue
            parts = []
            for item in items:
                tag_txt = ""
                if item["tags"]:
                    tag_txt = "・" + "・".join(item["tags"])

                parts.append(
                    f"{_fmt_line(item['line'])}［FR={item['fr']:.3f}{tag_txt}］"
                )

            note_sections.append(f"{zone_name}：" + "／".join(parts))

    else:
        note_sections.append("【ライン評価グループ】")
        note_sections.append("順流域：—")
        note_sections.append("渦域：—")
        note_sections.append("逆流域：—")

    note_sections.append("")

        # =========================================================
    # ラスト半周補正 表示
    # =========================================================
    try:
        _lh_bonus_map = globals().get("last_half_bonus_map", {})
        _lh_reason_map = globals().get("last_half_reason_map", {})
        _before_map = globals().get("score_map_before_last_half", {})
        _after_map = globals().get("score_map_last_half_applied", {})

        if isinstance(_lh_bonus_map, dict) and _lh_bonus_map:
            note_sections.append("【ラスト半周補正】")

            _lh_pairs = sorted(
                [(int(k), float(v)) for k, v in _lh_bonus_map.items()],
                key=lambda t: t[0]
            )

            for _car, _bonus in _lh_pairs:
                _before = float(_before_map.get(_car, 0.0) or 0.0)
                _after = float(_after_map.get(_car, _before + _bonus) or 0.0)

                _reasons = _lh_reason_map.get(_car, [])
                if not isinstance(_reasons, list):
                    _reasons = [_reasons]

                _reason_txt = "／".join(str(x) for x in _reasons if str(x).strip())
                if not _reason_txt:
                    _reason_txt = "補正なし"

                note_sections.append(
                    f"{_car}：展開={_before:.6f} ／ 補正={_bonus:+.3f} ／ 最終={_after:.6f}［{_reason_txt}］"
                )

            note_sections.append("")

    except Exception as _e:
        note_sections.append(f"※ラスト半周補正表示エラー：{_e}")
        note_sections.append("")
    # =========================================================
    # 会場×最終Hライン補正 表示
    # =========================================================
    try:
        _hf_bonus_map = globals().get("home_flow_bonus_map", {})
        _hf_reason_map = globals().get("home_flow_reason_map", {})
        _hf_before_map = globals().get("score_map_before_home_flow", {})
        _hf_after_map = globals().get("score_map_home_flow_applied", {})

        if isinstance(_hf_bonus_map, dict) and _hf_bonus_map:
            note_sections.append("【会場×最終Hライン補正】")
            note_sections.append(
                f"会場判定={globals().get('venue_profile', 'unknown')} ／ "
                f"補正倍率={float(globals().get('venue_home_flow_mult', 1.0)):.2f} ／ "
                f"必要オッズ倍率={float(globals().get('venue_min_odds_mult', 1.0)):.2f}"
            )

            _hf_pairs = sorted(
                [(int(k), float(v)) for k, v in _hf_bonus_map.items()],
                key=lambda t: t[0]
            )

            for _car, _bonus in _hf_pairs:
                _before = float(_hf_before_map.get(_car, 0.0) or 0.0)
                _after = float(_hf_after_map.get(_car, _before + _bonus) or 0.0)
                _reason_txt = str(_hf_reason_map.get(_car, ""))
                note_sections.append(
                    f"{_car}：補正前={_before:.6f} ／ H補正={_bonus:+.3f} ／ 補正後={_after:.6f}［{_reason_txt}］"
                )

            note_sections.append("")

    except Exception as _e:
        note_sections.append(f"※会場×最終H補正表示エラー：{_e}")
        note_sections.append("")

    # =========================================================
    # KO使用スコア（降順）
    # =========================================================
    _sc_pairs = sorted(
        [(int(k), float(v)) for k, v in (score_map or {}).items()],
        key=lambda t: (-t[1], t[0])
    )
    globals()["KO_SCORE_ORDER_FOR_TIE"] = [int(n) for n, _sc in _sc_pairs]
    globals()["KO_SCORE_MAP_FOR_SANTEN"] = {int(n): float(_sc) for n, _sc in _sc_pairs}

    note_sections.append("【KO使用スコア（降順）】")

    
    if _sc_pairs:
        for i, (n, sc) in enumerate(_sc_pairs, start=1):
            note_sections.append(f"{i}位：{n} (スコア={sc:.6f})")
    else:
        note_sections.append("—")
    note_sections.append("")

    # =========================================================
    # 最終ジャン想定隊列 → KO（6パターン）
    #   ワープ禁止：全体再ソート禁止
    #   距離：隣同士の交換のみ + 交換コスト
    #   重要：1パス中に同一車が何回も抜けない
    # =========================================================
    def _append_ko_queue_predictions(note_sections, all_lines, score_map, FR_line, VTX_line, U_line, _lfr):
        def _digits_of_line(ln):
            s = "".join(ch for ch in str(ln) if ch.isdigit())
            return [int(ch) for ch in s] if s else []

        def _norm_line(ln):
            return "".join(ch for ch in str(ln) if ch.isdigit())

        _PATTERNS = [
            ("順流→渦→逆流", ["順流", "渦", "逆流"]),
            ("順流→逆流→渦", ["順流", "逆流", "渦"]),
            ("渦→順流→逆流", ["渦", "順流", "逆流"]),
            ("渦→逆流→順流", ["渦", "逆流", "順流"]),
            ("逆流→順流→渦", ["逆流", "順流", "渦"]),
            ("逆流→渦→順流", ["逆流", "渦", "順流"]),
        ]

        def _infer_line_zone(ln):
            s = _norm_line(ln)

            # 新方式：ライン評価グループを優先
            try:
                zmap = globals().get("LINE_ZONE_MAP", {})
                if isinstance(zmap, dict) and s in zmap:
                    return zmap.get(s, "その他")
            except Exception:
                pass

            # 保険：旧方式
            if s and FR_line and s == _norm_line(FR_line):
                return "順流"
            if VTX_line and s == _norm_line(VTX_line):
                return "渦"
            if s and U_line and s == _norm_line(U_line):
                return "逆流"

            return "その他"

        def _queue_for_pattern(lines, svr_order):
            lines = list(lines or [])
            bucket = {"順流": [], "渦": [], "逆流": [], "その他": []}
            for ln in lines:
                bucket[_infer_line_zone(ln)].append(ln)

            queue = []
            for z in (svr_order or ["順流", "渦", "逆流"]):
                xs = sorted(bucket.get(z, []), key=lambda x: _lfr(x), reverse=True)
                for ln in xs:
                    queue.extend(_digits_of_line(ln))

            xs = sorted(bucket.get("その他", []), key=lambda x: _lfr(x), reverse=True)
            for ln in xs:
                queue.extend(_digits_of_line(ln))

            if not queue:
                for ln in lines:
                    queue.extend(_digits_of_line(ln))
            return queue

        def _build_car_zone_map(lines):
            m = {}
            for ln in (lines or []):
                z = _infer_line_zone(ln)
                for c in _digits_of_line(ln):
                    m[int(c)] = z
            return m

        _car_zone_map = _build_car_zone_map(all_lines)

        _car_line_size = {}
        _car_line_pos = {}

        for ln in (all_lines or []):
            ds = _digits_of_line(ln)
            sz = len(ds)

            for idx, c in enumerate(ds):
                _car_line_size[int(c)] = sz if sz > 0 else 1
                _car_line_pos[int(c)] = int(idx)

        def _pos_adj_for_car(car):
            """
            位置補正は隊列全体の何番目かではなく、
            その車が所属ライン内で何番手かを見る。
            単騎は番手利を与えない。
            """
            car = int(car)
            sz = int(_car_line_size.get(car, 1) or 1)
            pos = int(_car_line_pos.get(car, 0) or 0)

            # 単騎は位置補正なし
            if sz <= 1:
                return 0.0

            # ライン先頭
            if pos == 0:
                return -0.040

            # ライン2番手
            if pos == 1:
                return +0.020

            # 3番手以降
            return 0.0

        _FR_K_MAIN = 0.18
        _FR_K_SUB = 0.06
        _FR_BONUS_CAP = 0.06

        def _fr_bonus_for_car(car, main_zone):
            z = _car_zone_map.get(int(car), "その他")
            z_fr = {
                "順流": float(_lfr(FR_line) if FR_line else 0.0),
                "渦":   float(_lfr(VTX_line) if VTX_line else 0.0),
                "逆流": float(_lfr(U_line) if U_line else 0.0),
            }.get(z, 0.0)

            k = _FR_K_MAIN if z == main_zone else _FR_K_SUB
            sz = float(_car_line_size.get(int(car), 1) or 1.0)

            bonus = (k * z_fr) / sz
            if bonus > _FR_BONUS_CAP:
                bonus = _FR_BONUS_CAP
            if bonus < 0.0:
                bonus = 0.0
            return bonus

        def _run_ko(q, main_zone):
            # ======================================================
            # 距離ベース（B）＋ KO閾値（C）
            # ======================================================
            q = [int(x) for x in (q or []) if str(x).isdigit()]

            seen = set()
            order = []
            for c in q:
                if c not in seen:
                    seen.add(c)
                    order.append(c)

        def _run_ko(q, main_zone):
            # ======================================================
            # 距離ベース（B）＋ KO閾値（C）
            # ======================================================
            q = [int(x) for x in (q or []) if str(x).isdigit()]

            seen = set()
            order = []
            for c in q:
                if c not in seen:
                    seen.add(c)
                    order.append(c)


            tail = [int(c) for c in score_map.keys() if int(c) not in seen]
            tail.sort(key=lambda c: float(score_map.get(int(c), 0.0)), reverse=True)
            order.extend(tail)

            straight_m = float(globals().get("straight_length", 60.0) or 60.0)
            style = float(globals().get("style", 0.0) or 0.0)
            wind_ms = float(globals().get("wind_speed", 0.0) or 0.0)
            race_class = str(globals().get("race_class", "Ａ級") or "Ａ級")

            CLASS_SPREAD = {"Ｓ級": 1.00, "Ａ級": 0.90, "Ａ級チャレンジ": 0.80, "ガールズ": 0.85}
            spread = float(CLASS_SPREAD.get(race_class, 0.90))

            def _final_at(car, i):
                base = float(score_map.get(int(car), 0.0))
                return base + _pos_adj_for_car(int(car)) + _fr_bonus_for_car(int(car), main_zone)
            
            # ====== PATCH: venue-aware pass_m / available_m + speed-based MAX_PASSES ======
            pass_m = 14.0 + 0.35 * straight_m
            pass_m *= (1.0 + 0.25 * max(0.0, style))
            pass_m *= (1.0 + 0.03 * max(0.0, wind_ms - 3))

            # 会場カント（薄く：外回しロス増）
            bank_angle = float(globals().get("bank_angle", 30.0) or 30.0)
            pass_m *= (1.0 + 0.10 * max(0.0, (bank_angle - 30.0) / 10.0))  # 36°で+6%程度

            # クリップ
            if pass_m < 18.0:
                pass_m = 18.0
            if pass_m > 55.0:
                pass_m = 55.0

            # ---- available_m: bank_len を “差分だけ” 反映して飽和を減らす ----
            bank_len = float(globals().get("bank_length", 400.0) or 400.0)
            base_bank = 400.0
            # bank_len差分の反映を少し強める（500が1回に張り付くのを緩和）
            bank_term = 0.20 * base_bank + 0.30 * (bank_len - base_bank)
            available_m = float(straight_m) + bank_term

            # ---- スコア分布（sigma）----
            vals = [float(score_map.get(int(c), 0.0)) for c in order]
            if len(vals) >= 2:
                mu = sum(vals) / float(len(vals))
                var = sum((v - mu) ** 2 for v in vals) / float(len(vals))
                sigma = max(var ** 0.5, 1e-6)
            else:
                mu = (vals[0] if vals else 0.0)
                sigma = 1e-6

            # ---- クラス別の代表速度（終盤の代表値）----
            VREF_KMH = {"Ｓ級": 67.0, "Ａ級": 64.0, "Ａ級チャレンジ": 62.0, "ガールズ": 63.0}
            v_ref = float(VREF_KMH.get(race_class, 64.0)) / 3.6  # m/s

            # ---- スコア→速度：zで圧縮（暴走防止）----
            # 333/335は終盤時間が短く gain_m が出にくいので少し強める
            if bank_len <= 335:
                k_speed = float(globals().get("ko_k_speed_333", 0.014) or 0.014)
            elif bank_len >= 500:
                k_speed = float(globals().get("ko_k_speed_500", 0.012) or 0.012)
            else:
                k_speed = float(globals().get("ko_k_speed", 0.011) or 0.011)
            def _v_from_score(sc: float) -> float:
                z = (float(sc) - float(mu)) / float(sigma)
                if z > 2.0:
                    z = 2.0
                if z < -2.0:
                    z = -2.0
                return float(v_ref) * (1.0 + float(k_speed) * z)

            # ---- 終盤時間 & 相対距離（抜ける回数の根拠）----
            t_final = float(available_m) / max(float(v_ref), 1e-6)

            top_scores = sorted(vals, reverse=True)
            if len(top_scores) >= 3:
                v_fast = _v_from_score(top_scores[0])
                v_mid  = _v_from_score(top_scores[2])
            else:
                v_fast = _v_from_score(mu + sigma)
                v_mid  = _v_from_score(mu)

            gain_m = max(0.0, (float(v_fast) - float(v_mid)) * float(t_final))

            MAX_PASSES = int(gain_m // max(pass_m, 1e-9))
            if MAX_PASSES < 1:
                MAX_PASSES = 1

            # 333/335は最大2、その他は最大3（過剰シャッフル防止）
            cap = 2 if bank_len <= 335 else 3
            if MAX_PASSES > cap:
                MAX_PASSES = cap

            # ---- PASS_DELTAの正規化：available_m依存を弱めて安定化 ----
            base_k = float(globals().get("ko_base_k", 0.040) or 0.040)  # 0.025〜0.060
            score_per_m = base_k * sigma * (1.0 / max(spread, 1e-6)) / max(pass_m, 1e-6)

            PASS_DELTA = score_per_m * pass_m
            cross_mul = 0.35 if bank_len <= 335 else 0.30
            CROSS_DELTA = score_per_m * (cross_mul * pass_m)
            fatigue_delta = 0.35 * PASS_DELTA
            # ====== /PATCH ======

            overtake_cnt = {int(c): 0 for c in order}

            for _ in range(MAX_PASSES):
                swapped = False
                n = len(order)
                moved_this_pass = set()

                for i in range(n - 1):
                    a = order[i]
                    b = order[i + 1]

                    if b in moved_this_pass:
                        continue

                    sa = _final_at(a, i)
                    sb = _final_at(b, i + 1)

                    need = PASS_DELTA + fatigue_delta * float(overtake_cnt.get(b, 0))

                    za = _car_zone_map.get(int(a), "その他")
                    zb = _car_zone_map.get(int(b), "その他")
                    if za != zb:
                        need += CROSS_DELTA

                    if sb >= sa + need:
                        order[i], order[i + 1] = b, a
                        overtake_cnt[b] = overtake_cnt.get(b, 0) + 1
                        moved_this_pass.add(b)
                        swapped = True

                if not swapped:
                    break

            globals()["_overtake_available_m"] = float(available_m)
            globals()["_overtake_pass_m"] = float(pass_m)
            globals()["_overtake_max_passes"] = int(MAX_PASSES)
            globals()["_overtake_pass_delta"] = float(PASS_DELTA)
            globals()["_overtake_cross_delta"] = float(CROSS_DELTA)

            # 任意：調整が速くなるデバッグ（欲しければ d 表示にも足せる）
            globals()["_overtake_gain_m"] = float(gain_m)
            globals()["_overtake_t_final"] = float(t_final)
            globals()["_overtake_v_ref"] = float(v_ref)

            return order

            globals()["_overtake_available_m"] = float(available_m)
            globals()["_overtake_pass_m"] = float(pass_m)
            globals()["_overtake_max_passes"] = int(MAX_PASSES)
            globals()["_overtake_pass_delta"] = float(PASS_DELTA)
            globals()["_overtake_cross_delta"] = float(CROSS_DELTA)

            return order

        outs = {}
        for pname, svr in _PATTERNS:
            q = _queue_for_pattern(all_lines, svr)
            main_zone = (svr[0] if (svr and len(svr) >= 1) else "順流")
            outs[pname] = _run_ko(q, main_zone)

        def _fmt_seq(seq, max_n=None):
            xs = [int(x) for x in (seq or []) if str(x).isdigit()]
            if max_n is None:
                max_n = int(globals().get("n_cars", len(xs)))
            xs = xs[:max_n]
            return " → ".join(str(x) for x in xs) if xs else "（なし）"

        out_j = outs.get("順流→渦→逆流") or []
        out_v = outs.get("渦→順流→逆流") or []
        out_u = outs.get("逆流→順流→渦") or []

        

                       # ======================================================
        # 表示用ガード：
        # 1) KO隊列結果がスコア下位を頭に置きすぎる場合だけ補正
        # 2) 主戦ライン先頭が同ライン低スコア車より後ろに落ちるのを防ぐ
        # ※ _run_ko本体は触らない
        # ======================================================
        def _digits_line(x):
            return [int(ch) for ch in str(x) if ch.isdigit()]

        def _display_score_guard(seq, main_line=None):
            xs = [int(x) for x in (seq or []) if str(x).isdigit()]
            if not xs:
                return xs

            score_order = sorted(
                [int(k) for k in score_map.keys()],
                key=lambda c: (-float(score_map.get(c, 0.0)), c)
            )
            score_rank = {c: i + 1 for i, c in enumerate(score_order)}

            # 1) 先頭ガード
            # 先頭がKOスコア5位以下なら、スコア上位3台のうち
            # 元の隊列内で一番前にいる車を先頭へ上げる
            head = xs[0]
            if score_rank.get(head, 99) >= 5:
                candidates = [c for c in score_order[:3] if c in xs]
                if candidates:
                    best = min(candidates, key=lambda c: xs.index(c))
                    xs.remove(best)
                    xs.insert(0, best)

                        # 2) 主戦ライン先頭ガード
            # 例：364なら3がライン先頭。
            # 3よりスコアが低い同ライン車（例：6）が3より前にいるなら、
            # 3をその車の前まで戻す。
            line_members = _digits_line(main_line)
            if len(line_members) >= 2:
                line_head = line_members[0]

                if line_head in xs:
                    line_head_score = float(score_map.get(line_head, 0.0))
                    line_head_idx = xs.index(line_head)

                    lower_mates_before = []
                    for m in line_members[1:]:
                        if m in xs:
                            m_score = float(score_map.get(m, 0.0))
                            if m_score < line_head_score and xs.index(m) < line_head_idx:
                                lower_mates_before.append(m)

                    if lower_mates_before:
                        target_idx = min(xs.index(m) for m in lower_mates_before)
                        xs.remove(line_head)
                        xs.insert(target_idx, line_head)

                        # 3) 最下位スコア車の早出しガード
            # KOスコア最下位の車が3番手以内に残るのを防ぐ
            n_score = len(score_order)

            for bad in list(xs):
                if score_rank.get(bad, 99) == n_score and xs.index(bad) <= 2:
                    xs.remove(bad)

                    # スコア5位以内の車が並んだ最後の直後へ送る
                    insert_pos = 0
                    for i, c in enumerate(xs):
                        if score_rank.get(c, 99) <= 5:
                            insert_pos = i + 1

                    xs.insert(insert_pos, bad)

                        # 4) KO上位車の沈みすぎガード
            # KOスコア上位3車が沈みすぎるのを防ぐ
            # 1位は頭候補、2〜3位は3番手以内を目安に戻す
            for good in score_order[:3]:
                if good not in xs:
                    continue

                r = score_rank.get(good, 99)

                # KO2〜3位が4番手以下なら、3番手以内へ戻す
                if r in (2, 3) and xs.index(good) >= 3:
                    xs.remove(good)
                    target_pos = min(2, len(xs))
                    xs.insert(target_pos, good)

                # KO1位が3番手以下なら、2番手以内へ戻す
                elif r == 1 and xs.index(good) >= 2:
                    xs.remove(good)
                    target_pos = min(1, len(xs))
                    xs.insert(target_pos, good)

            return xs

        # ======================================================
        # v195：戦法別シナリオ補正
        # 順流・渦・逆流は「同じ全体順位の別名」ではなく、
        # それぞれの流域ラインが主役になった場合の着順予想として組み立てる。
        #
        # 重要：
        # ・無条件にライン先頭を1着固定するのではなく、ライン内のKO/役割で頭候補を選ぶ。
        # ・ただし、その戦法のシナリオでは主役ラインのいずれかが1着候補になる前提を守る。
        # ・逆流域がLINE_ZONE_MAP上で空でも、旧U_lineがあれば逆流シナリオの主役ラインとして使う。
        # ======================================================
        def _scenario_line_digits(_ln):
            try:
                if isinstance(_ln, (list, tuple)):
                    return [int(x) for x in _ln if str(x).isdigit()]
            except Exception:
                pass
            return [int(ch) for ch in str(_ln) if ch.isdigit()]

        def _scenario_line_key(_ln):
            return "".join(str(int(x)) for x in _scenario_line_digits(_ln))

        def _scenario_same_line(_a, _b):
            return _scenario_line_key(_a) == _scenario_line_key(_b) and bool(_scenario_line_key(_a))

        def _scenario_lines_for_zone(_zone_name):
            """LINE_ZONE_MAPから該当ゾーンのラインを取得。line_defを優先して実ライン順を復元する。"""
            out_lines = []
            seen_keys = set()
            try:
                zmap = globals().get("LINE_ZONE_MAP", {}) or {}
                _line_def_local = globals().get("line_def", {}) or {}
                if isinstance(_line_def_local, dict):
                    for _gid, _mem in _line_def_local.items():
                        _xs = _scenario_line_digits(_mem)
                        _key = _scenario_line_key(_xs)
                        if not _key or _key in seen_keys:
                            continue
                        if str(zmap.get(_key, "")) == str(_zone_name):
                            seen_keys.add(_key)
                            out_lines.append(_xs)
                # line_defに無いキーがあれば保険で拾う
                if isinstance(zmap, dict):
                    for _key, _z in zmap.items():
                        if str(_z) != str(_zone_name):
                            continue
                        if str(_key) in seen_keys:
                            continue
                        _xs = _scenario_line_digits(_key)
                        if _xs:
                            seen_keys.add(str(_key))
                            out_lines.append(_xs)
            except Exception:
                out_lines = []
            try:
                out_lines = sorted(out_lines, key=lambda _ln: float(_lfr(_ln)), reverse=True)
            except Exception:
                pass
            return out_lines

        def _scenario_main_line(_style_name):
            """
            各戦法の主役ライン。

            v197:
            2ライン戦などで、渦と逆流が同じラインを主役にして
            同じ買目考察を二重表示するのを禁止する。
            逆流は「LINE_ZONE_MAP上の逆流域」または「旧U_line」が、
            順流/渦の主役ラインと別ラインとして存在する場合だけ採用する。
            """
            try:
                fr_key = _scenario_line_key(FR_line) if FR_line else ""
                vtx_key = _scenario_line_key(VTX_line) if VTX_line else ""

                if _style_name == "順流":
                    if FR_line:
                        return _scenario_line_digits(FR_line)
                    _ls = _scenario_lines_for_zone("順流")
                    return _ls[0] if _ls else []

                if _style_name == "渦":
                    if VTX_line:
                        return _scenario_line_digits(VTX_line)
                    _ls = _scenario_lines_for_zone("渦")
                    return _ls[0] if _ls else []

                if _style_name == "逆流":
                    used_keys = {k for k in (fr_key, vtx_key) if k}

                    # まずLINE_ZONE_MAP上で明示された逆流域を優先。
                    # ただし順流/渦の主役ラインと同一なら採用しない。
                    _ls = _scenario_lines_for_zone("逆流")
                    for _ln in (_ls or []):
                        _key = _scenario_line_key(_ln)
                        if _key and _key not in used_keys:
                            return _scenario_line_digits(_ln)

                    # 逆流域が空の場合のみ、旧U_lineを補完候補にする。
                    # ただし旧U_lineが渦ライン等と同一なら、逆流を無理に作らない。
                    if U_line:
                        u_key = _scenario_line_key(U_line)
                        if u_key and u_key not in used_keys:
                            return _scenario_line_digits(U_line)

                    return []
            except Exception:
                return []
            return []

        def _scenario_queue_for_main(_main_line, _zone_order):
            """主役ラインを先頭に置いた仮想隊列。残りはゾーン順＋FR順で並べる。"""
            main = _scenario_line_digits(_main_line)
            main_key = _scenario_line_key(main)
            queue = []
            seen_cars = set()

            for c in main:
                if int(c) not in seen_cars:
                    seen_cars.add(int(c))
                    queue.append(int(c))

            # all_linesが取れる場合はライン単位、無い場合は既存のSTYLE_SEQ_MAP相当で補完。
            used_line_keys = {main_key} if main_key else set()
            try:
                lines_src = list(all_lines or [])
            except Exception:
                lines_src = []

            bucket = {"順流": [], "渦": [], "逆流": [], "その他": []}
            for ln in lines_src:
                key = _scenario_line_key(ln)
                if key and key in used_line_keys:
                    continue
                z = _infer_line_zone(ln)
                # U_lineは逆流シナリオでは既に主役として使うため、渦側に重複させない。
                bucket.setdefault(z, []).append(ln)

            for z in (_zone_order or []):
                xs = sorted(bucket.get(z, []), key=lambda ln: float(_lfr(ln)), reverse=True)
                for ln in xs:
                    key = _scenario_line_key(ln)
                    if key and key in used_line_keys:
                        continue
                    used_line_keys.add(key)
                    for c in _scenario_line_digits(ln):
                        if int(c) not in seen_cars:
                            seen_cars.add(int(c))
                            queue.append(int(c))

            for z in ["順流", "渦", "逆流", "その他"]:
                if z in (_zone_order or []):
                    continue
                xs = sorted(bucket.get(z, []), key=lambda ln: float(_lfr(ln)), reverse=True)
                for ln in xs:
                    key = _scenario_line_key(ln)
                    if key and key in used_line_keys:
                        continue
                    used_line_keys.add(key)
                    for c in _scenario_line_digits(ln):
                        if int(c) not in seen_cars:
                            seen_cars.add(int(c))
                            queue.append(int(c))

            # 保険：score_mapに存在する車を全て補完
            try:
                tail = sorted([int(c) for c in score_map.keys() if int(c) not in seen_cars], key=lambda c: float(score_map.get(c, 0.0)), reverse=True)
                queue.extend(tail)
            except Exception:
                pass
            return queue

        def _scenario_best_head_from_main_line(_main_line):
            """主役ライン内で最も頭に置きやすい車を選ぶ。ライン先頭固定ではない。"""
            main = _scenario_line_digits(_main_line)
            if not main:
                return None
            def _role_bonus(_car):
                try:
                    pos = main.index(int(_car))
                except Exception:
                    pos = 0
                # 先頭と番手を主に見る。3番手以降は頭固定しにくいが、KOが抜けていれば上がれる。
                if len(main) <= 1:
                    return 0.00
                if pos == 0:
                    return 0.035
                if pos == 1:
                    return 0.025
                return -0.015
            try:
                return max(main, key=lambda c: float(score_map.get(int(c), 0.0)) + _role_bonus(c))
            except Exception:
                return int(main[0])

        def _scenario_force_main_head(_seq, _main_line):
            """
            シナリオの前提として、主役ラインのいずれかを1着候補へ置く。

            v196:
            1着候補だけを先頭へ上げても、同ライン相手が後方へ沈むと
            「そのラインが主役になった展開」として買目妙味が効かない。
            そのため、主役ラインの残りも2〜4番手以内へ保護する。
            ただしライン丸ごと無条件固定ではなく、KO/既存順位を見て並べる。
            """
            xs = [int(x) for x in (_seq or []) if str(x).isdigit()]
            main = _scenario_line_digits(_main_line)
            if not xs or not main:
                return xs

            main_set = {int(c) for c in main}
            head = int(xs[0]) if int(xs[0]) in main_set else _scenario_best_head_from_main_line(main)
            if head is None or int(head) not in xs:
                return xs
            head = int(head)

            # 主役ライン内の残りは、KOスコア＋既存順位で2〜4番手へ寄せる。
            # 2車ラインなら相手を2番手へ、3車以上なら最大2車までを上位保護する。
            main_rest = [int(c) for c in main if int(c) != head and int(c) in xs]
            try:
                rank_now = {int(c): i for i, c in enumerate(xs)}
                main_rest = sorted(
                    main_rest,
                    key=lambda c: (float(score_map.get(int(c), 0.0)), -int(rank_now.get(int(c), 99))),
                    reverse=True,
                )
            except Exception:
                pass

            protect_count = 1 if len(main) <= 2 else 2
            protected = [head] + main_rest[:protect_count]

            out = []
            seen = set()
            for c in protected:
                c = int(c)
                if c in xs and c not in seen:
                    out.append(c)
                    seen.add(c)

            for c in xs:
                c = int(c)
                if c not in seen:
                    out.append(c)
                    seen.add(c)
            return out

        def _make_style_scenario_seq(_style_name, _fallback_seq):
            main_line = _scenario_main_line(_style_name)
            if not main_line:
                # v197:
                # シナリオ主役ラインが存在しない流れは、
                # fallbackで無理に買目考察を作らない。
                # 例：2ライン戦で渦=旧逆流ラインの場合、逆流は空扱い。
                return []

            if _style_name == "順流":
                zone_order = ["順流", "渦", "逆流"]
                fallback_main = FR_line
            elif _style_name == "渦":
                zone_order = ["渦", "順流", "逆流"]
                fallback_main = VTX_line
            else:
                zone_order = ["逆流", "順流", "渦"]
                fallback_main = U_line

            q = _scenario_queue_for_main(main_line, zone_order)
            seq = _run_ko(q, _style_name)
            seq = _display_score_guard(seq, main_line or fallback_main)
            seq = _scenario_force_main_head(seq, main_line)
            return [int(x) for x in (seq or []) if str(x).isdigit()]

        # 旧表示順は保持しておく。以後の買目考察にはシナリオ補正版を使う。
        out_j_raw = _display_score_guard(out_j, FR_line)
        out_v_raw = _display_score_guard(out_v, VTX_line)
        out_u_raw = _display_score_guard(out_u, U_line)

        out_j = _make_style_scenario_seq("順流", out_j_raw)
        out_v = _make_style_scenario_seq("渦", out_v_raw)
        out_u = _make_style_scenario_seq("逆流", out_u_raw)

        globals()["STYLE_BASE_SEQ_MAP"] = {
            "順流": [int(x) for x in (out_j_raw or []) if str(x).isdigit()],
            "渦":   [int(x) for x in (out_v_raw or []) if str(x).isdigit()],
            "逆流": [int(x) for x in (out_u_raw or []) if str(x).isdigit()],
        }
        globals()["STYLE_SCENARIO_MAIN_LINE_MAP"] = {
            "順流": _scenario_main_line("順流"),
            "渦":   _scenario_main_line("渦"),
            "逆流": _scenario_main_line("逆流"),
        }

        # ======================================================
        # H主導ライン3番手以降：
        # 3着内率40%以上なら、
        # 「その戦法の表示1着候補ライン」と同じ場合だけ4番手以内へ移動
        # ======================================================
        try:
            def _display_promote_gid(_car_no):
                try:
                    _car_no = int(_car_no)
                    if isinstance(line_def, dict):
                        for _gid, _mem in line_def.items():
                            _mem2 = [int(x) for x in _mem]
                            if _car_no in _mem2:
                                return _gid
                except Exception:
                    pass
                return None

            def _display_promote_top3_rate(_car_no):
                try:
                    _car_no = int(_car_no)

                    _x1 = globals().get("x1", {})
                    _x2 = globals().get("x2", {})
                    _x3 = globals().get("x3", {})
                    _xo = globals().get("x_out", {})

                    n1 = float(_x1.get(_car_no, _x1.get(str(_car_no), 0)) or 0)
                    n2 = float(_x2.get(_car_no, _x2.get(str(_car_no), 0)) or 0)
                    n3 = float(_x3.get(_car_no, _x3.get(str(_car_no), 0)) or 0)
                    no = float(_xo.get(_car_no, _xo.get(str(_car_no), 0)) or 0)

                    total = n1 + n2 + n3 + no
                    if total <= 0:
                        return None

                    return float((n1 + n2 + n3) / total)

                except Exception:
                    return None

            def _display_promote_to_top4(_seq, _target_car):
                try:
                    _target_car = int(_target_car)
                    _xs = [int(x) for x in (_seq or []) if str(x).isdigit()]

                    if _target_car not in _xs:
                        return _xs

                    _idx = _xs.index(_target_car)

                    # すでに4番手以内なら何もしない
                    if _idx <= 3:
                        return _xs

                    _xs.pop(_idx)
                    _xs.insert(3, _target_car)

                    return _xs

                except Exception:
                    return _seq

            # H主導ラインの3番手以降で、3着内率40%以上の車だけ対象
            _promote_targets = []

            if home_top_gid is not None and isinstance(line_def, dict):
                _h_members = [int(x) for x in line_def.get(home_top_gid, [])]

                if len(_h_members) >= 3:
                    for _car3 in _h_members[2:]:
                        _p3 = _display_promote_top3_rate(_car3)

                        if _p3 is not None and float(_p3) >= 0.40:
                            _promote_targets.append(int(_car3))

            # 各戦法の「表示上の1着候補ライン」と同じ場合だけ、4番手以内へ移動
            for _car3 in _promote_targets:
                _target_gid = _display_promote_gid(_car3)

                if _target_gid is None:
                    continue

                # 順流
                if out_j:
                    _jun_head = int(out_j[0])
                    _jun_gid = _display_promote_gid(_jun_head)
                    if _target_gid == _jun_gid:
                        out_j = _display_promote_to_top4(out_j, _car3)

                # 渦
                if out_v:
                    _vtx_head = int(out_v[0])
                    _vtx_gid = _display_promote_gid(_vtx_head)
                    if _target_gid == _vtx_gid:
                        out_v = _display_promote_to_top4(out_v, _car3)

                # 逆流
                if out_u:
                    _u_head = int(out_u[0])
                    _u_gid = _display_promote_gid(_u_head)
                    if _target_gid == _u_gid:
                        out_u = _display_promote_to_top4(out_u, _car3)

        except Exception as _e:
            note_sections.append(f"※H主導3番手以降・戦法別4番手以内補正エラー：{_e}")
            note_sections.append("")

        # ======================================================
        # 戦法別評価順を保存
        # v195以降：STYLE_SEQ_MAPは、流域ライン主役のシナリオ補正版を保存する。
        # 後段の「戦法別想定決着率」「2車複候補」「買目考察」はこの補正版を使う。
        # 元の全体KO寄り順位は STYLE_BASE_SEQ_MAP に保持済み。
        # ======================================================
        globals()["STYLE_SCENARIO_SEQ_MAP"] = {
            "順流": [int(x) for x in (out_j or []) if str(x).isdigit()],
            "渦":   [int(x) for x in (out_v or []) if str(x).isdigit()],
            "逆流": [int(x) for x in (out_u or []) if str(x).isdigit()],
        }
        globals()["STYLE_SEQ_MAP"] = dict(globals().get("STYLE_SCENARIO_SEQ_MAP", {}) or {})

        # ======================================================
        # 戦法別着順予想を全表示
        # ※ここでは推奨戦法がまだ確定していないため、強調はしない。
        #   後段で「推奨戦法＋コピー用」を別枠表示する。
        # ======================================================
        try:
            def _fmt_seq_full(_seq):
                _xs = [int(x) for x in (_seq or []) if str(x).isdigit()]
                return " → ".join(str(x) for x in _xs) if _xs else "該当なし"

            note_sections.append("【順流メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_j))
            note_sections.append("")
            note_sections.append("【渦メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_v))
            note_sections.append("")
            note_sections.append("【逆流メイン着順予想】")
            note_sections.append(_fmt_seq_full(out_u))
            note_sections.append("")
        except Exception as _e:
            note_sections.append(f"※戦法別着順予想表示エラー：{_e}")
            note_sections.append("")


    _append_ko_queue_predictions(note_sections, all_lines, score_map, FR_line, VTX_line, U_line, _lfr)
    # ここまでで note_sections を確実に保持

        # =========================================================
    # ＜短評＞（KOの成否に関係なく表示）※完全tryゼロ
    # =========================================================
    lines_out = ["＜短評＞"]

    # レースFR：flowのFR（過去出力と同じ定義）
    raceFR = float(_flow.get("FR", 0.0) or 0.0) if isinstance(_flow, dict) else 0.0
    if raceFR != raceFR:  # NaN
        raceFR = 0.0

    # flowが0なら「混戦度」= 1 - 最大取り分（line_fr_mapがあれば）
    if raceFR <= 0.0 and isinstance(line_fr_map, dict) and line_fr_map:
        vals = []
        for v in line_fr_map.values():
            s = str(v).strip()
            fv = float(s) if s not in ("", "None", "nan", "NaN") else 0.0
            if fv > 0.0 and fv == fv:
                vals.append(fv)

        total = sum(vals)
        if total > 1e-12:
            max_share = max(fv / total for fv in vals)
            raceFR = 1.0 - max_share
            if raceFR < 0.0:
                raceFR = 0.0
            if raceFR > 1.0:
                raceFR = 1.0

    # レースFR表示
    lines_out.append(f"・レースFR={raceFR:.3f}［{_band3_fr(raceFR)}］")

    # 混戦度表示
    _compact_label = globals().get("race_compact_label", "未判定")
    _compact_gap = globals().get("race_compact_gap", None)

    if _compact_gap is not None:
        lines_out.append(
            f"・順当度：{_compact_label}［上位差={float(_compact_gap):.2f}］"
        )
    else:
        lines_out.append(
            f"・順当度：{_compact_label}"
        )

    # VTX/U はラインFR（ズレ防止）
    _vtx_fr = float(_lfr(VTX_line) if VTX_line else 0.0)
    _u_fr = float(_lfr(U_line) if U_line else 0.0)

    

    lines_out.append(f"・VTXラインFR={_vtx_fr:.3f}［{_band3_vtx(_vtx_fr)}］")
    lines_out.append(f"・逆流ラインFR={_u_fr:.3f}［{_band3_u(_u_fr)}］")

    # 内訳要約（flow dbg）
    dbg = _flow.get("dbg", {}) if isinstance(_flow, dict) else {}

    if isinstance(dbg, dict) and dbg:
        bs = float(dbg.get("blend_star", 0.0) or 0.0)
        bn = float(dbg.get("blend_none", 0.0) or 0.0)
        sd = float(dbg.get("sd", 0.0) or 0.0)
        nu = float(dbg.get("nu", 0.0) or 0.0)

        star_txt = "先頭負担:強" if bs <= -0.60 else (
                   "先頭負担:中" if bs <= -0.30 else
                   "先頭負担:小")

        none_txt = "無印押上げ:強" if bn >= 1.20 else (
                   "無印押上げ:中" if bn >= 0.60 else
                   "無印押上げ:小")

        sd_txt = "ライン偏差:大" if sd >= 0.60 else (
                 "ライン偏差:中" if sd >= 0.30 else
                 "ライン偏差:小")

        nu_txt = "正規化:小" if 0.90 <= nu <= 1.10 else "正規化:補正強"

        lines_out.append(
            f"・内訳要約：{star_txt}／{none_txt}／{sd_txt}／{nu_txt}"
        )

    # =========================================================
    # ＜短評＞（KOの成否に関係なく表示）
    # =========================================================
    lines_out = ["＜短評＞"]

    raceFR = float(_flow.get("FR", 0.0) or 0.0) if isinstance(_flow, dict) else 0.0
    if raceFR != raceFR:
        raceFR = 0.0

    if raceFR <= 0.0 and isinstance(line_fr_map, dict) and line_fr_map:
        vals = []
        for v in line_fr_map.values():
            s = str(v).strip()
            fv = float(s) if s not in ("", "None", "nan", "NaN") else 0.0
            if fv > 0.0 and fv == fv:
                vals.append(fv)

        total = sum(vals)
        if total > 1e-12:
            max_share = max(fv / total for fv in vals)
            raceFR = 1.0 - max_share
            raceFR = max(0.0, min(1.0, raceFR))

        lines_out.append(f"・レースFR={raceFR:.3f}［{_band3_fr(raceFR)}］")

    # レースレベル表示
    try:
        lines_out.append(
            f"・レースレベル：{race_level_label}［平均得点={race_level_avg:.2f}／得点差={race_level_spread:.2f}］"
        )
    except Exception:
        pass

    _vtx_fr = float(_lfr(VTX_line) if VTX_line else 0.0)
    _u_fr = float(_lfr(U_line) if U_line else 0.0)

        # 混戦度表示
    _compact_label = globals().get("race_compact_label", "未判定")
    _compact_gap = globals().get("race_compact_gap", None)

    if _compact_gap is not None:
        lines_out.append(
            f"・順当度：{_compact_label}［上位差={float(_compact_gap):.2f}］"
        )
    else:
        lines_out.append(
            f"・順当度：{_compact_label}"
        )

    lines_out.append(f"・VTXラインFR={_vtx_fr:.3f}［{_band3_vtx(_vtx_fr)}］")
    lines_out.append(f"・逆流ラインFR={_u_fr:.3f}［{_band3_u(_u_fr)}］")

    bs = 0.0
    bn = 0.0
    sd = 0.0
    nu = 1.0

    dbg = _flow.get("dbg", {}) if isinstance(_flow, dict) else {}
    if isinstance(dbg, dict) and dbg:
        bs = float(dbg.get("blend_star", 0.0) or 0.0)
        bn = float(dbg.get("blend_none", 0.0) or 0.0)
        sd = float(dbg.get("sd", 0.0) or 0.0)
        nu = float(dbg.get("nu", 1.0) or 1.0)

    star_txt = "先頭負担:強" if bs <= -0.60 else ("先頭負担:中" if bs <= -0.30 else "先頭負担:小")
    none_txt = "無印押上げ:強" if bn >= 1.20 else ("無印押上げ:中" if bn >= 0.60 else "無印押上げ:小")
    sd_txt = "ライン偏差:大" if sd >= 0.60 else ("ライン偏差:中" if sd >= 0.30 else "ライン偏差:小")
    nu_txt = "正規化:小" if 0.90 <= nu <= 1.10 else "正規化:補正強"

    lines_out.append(f"・内訳要約：{star_txt}／{none_txt}／{sd_txt}／{nu_txt}")

    # =========================================================
    # 推奨戦法（優先順位固定・上書き禁止）
    # =========================================================

    try:
        recommend_style = None
        recommend_reason = []
        confidence = "C"

        tenkai_txt = str(
            globals().get("展開評価", "")
            or globals().get("tenkai_eval", "")
            or ""
        ).strip()

        fr_diff = abs(_vtx_fr - _u_fr)

                # =====================================================
        # 現在のライン評価グループでH主導ラインを判定する
        #   旧FR_line / 旧VTX_line / 旧U_line ではなく、
        #   LINE_ZONE_MAP を優先する
        # =====================================================

        def _norm_line_key_for_recommend(ln):
            try:
                if isinstance(ln, (list, tuple)):
                    return "".join(str(int(x)) for x in ln if str(x).isdigit())
            except Exception:
                pass
            return "".join(ch for ch in str(ln) if ch.isdigit())

        def _current_zone_for_line(ln):
            key = _norm_line_key_for_recommend(ln)

            try:
                zmap = globals().get("LINE_ZONE_MAP", {})
                if isinstance(zmap, dict) and key in zmap:
                    return zmap.get(key, "その他")
            except Exception:
                pass

            # 保険：LINE_ZONE_MAPが無い場合だけ旧方式へフォールバック
            if key and key == _norm_line_key_for_recommend(FR_line):
                return "順流"
            if key and key == _norm_line_key_for_recommend(VTX_line):
                return "渦"
            if key and key == _norm_line_key_for_recommend(U_line):
                return "逆流"

            return "その他"

        def _style_fr_for_recommend(style_name):
            if style_name == "順流":
                return float(_lfr(FR_line) if FR_line else 0.0)
            if style_name == "渦":
                return float(_lfr(VTX_line) if VTX_line else 0.0)
            if style_name == "逆流":
                return float(_lfr(U_line) if U_line else 0.0)
            return 0.0

        # =====================================================
        # 1. 展開評価（最優先）
        # =====================================================

        if "混戦" in tenkai_txt:
            recommend_style = "渦"
            recommend_reason = ["展開=混戦"]

        elif "差し" in tenkai_txt:
            recommend_style = "渦"
            recommend_reason = ["展開=差し寄り"]

        elif "先行" in tenkai_txt or "逃げ" in tenkai_txt:
            recommend_style = "順流"
            recommend_reason = ["展開=先行寄り"]

        # =====================================================
        # 2. 短評（ここで確定させる）
        # =====================================================

        if recommend_style is None:

            if bn >= 0.50:
                recommend_style = "渦"
                recommend_reason = ["無印押上げ=中以上"]

            elif sd >= 0.60:
                recommend_style = "順流"
                recommend_reason = ["ライン偏差=大"]

            elif bs <= -0.60 and bn >= 0.50:
                recommend_style = "逆流"
                recommend_reason = ["先頭負担強＋押上げ中以上"]

        # =====================================================
        # 3. FR差（ここは最後）
        # =====================================================

        if recommend_style is None:

            if fr_diff >= 0.02:

                if _u_fr > _vtx_fr:
                    recommend_style = "逆流"
                    recommend_reason = ["逆流FR優勢"]

                else:
                    recommend_style = "順流"
                    recommend_reason = ["VTX優勢"]

        # =====================================================
        # 4. 最終安全側
        # =====================================================

        if recommend_style is None:
            recommend_style = "渦"
            recommend_reason = ["標準判定"]

               
                
                # =====================================================
        # H：推奨理由への反映
        #   旧分類ではなく、現在のライン評価グループで判定
        # =====================================================
        try:
            if home_top_line == "主導なし":
                recommend_reason.append("H主導ラインなし")
            else:
                h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                h_zone = _current_zone_for_line(h_line)

                if h_zone in ("順流", "渦", "逆流"):
                    recommend_reason.append(f"H主導={h_zone}ライン")
                else:
                    recommend_reason.append("H主導=その他ライン")
        except Exception:
            pass

                
               
                # =====================================================
        # 信頼度
        # =====================================================
        if bn >= 0.50:
            confidence = "B"

        elif fr_diff >= 0.02:
            confidence = "A"

        elif fr_diff >= 0.01:
            confidence = "B"

        else:
            confidence = "C"

                # =====================================================
        # H：低信頼時の推奨戦法切り替え
        #   旧分類ではなく、現在のライン評価グループで判定
        #   ※ガールズはライン戦ではないため、H主導で戦法を切り替えない
        # =====================================================
        h_style = None
        h_changed = False

        try:
            if home_top_line != "主導なし":
                h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                h_zone = _current_zone_for_line(h_line)

                if h_zone in ("順流", "渦", "逆流"):
                    h_style = h_zone
                    h_fr = float(_lfr(h_line) if h_line else 0.0)
                else:
                    h_style = None
                    h_fr = 0.0

                cur_fr = _style_fr_for_recommend(recommend_style)

                if not is_girls_like:
                    if (
                        h_style is not None
                        and h_style != recommend_style
                        and confidence in ("B", "C")
                        and h_fr >= cur_fr - 0.01
                    ):
                        recommend_reason.append(f"H主導により{h_style}寄せ")
                        recommend_style = h_style
                        h_changed = True
                        confidence = "B"
                else:
                    recommend_reason.append("ガールズ/アドバンスのためH主導による戦法変更なし")
        except Exception:
            pass

        # =====================================================
        # H：信頼度への反映
        #   旧分類ではなく、現在のライン評価グループで判定
        #   ※ガールズはライン戦ではないため、H主導で信頼度も上下させない
        # =====================================================
        try:
            if not is_girls_like:
                if home_top_line != "主導なし":
                    h_line = line_def.get(home_top_gid, []) if home_top_gid is not None else []
                    h_zone = _current_zone_for_line(h_line)

                    h_match = (
                        h_zone in ("順流", "渦", "逆流")
                        and h_zone == recommend_style
                    )

                    h_conflict = (
                        h_zone in ("順流", "渦", "逆流")
                        and h_zone != recommend_style
                    )

                    if h_match:
                        if confidence == "C":
                            confidence = "B"
                        elif confidence == "B":
                            confidence = "A"

                    elif h_conflict:
                        if confidence == "A":
                            confidence = "B"
                        elif confidence == "B":
                            confidence = "C"

        except Exception:
            pass

        # Hで戦法変更した場合は、過信防止で信頼度AをBに抑える
        try:
            if h_changed and confidence == "A":
                confidence = "B"
        except Exception:
            pass

        # H反映チェック表示
        try:
            if h_style is not None:
                if h_changed:
                    recommend_reason.append("H反映=戦法変更あり")
                else:
                    recommend_reason.append("H反映=戦法変更なし")
        except Exception:
            pass

        # =====================================================
        # v167: KO上位3車の所属流域で推奨流れを補正
        #   ・順流域が2車以上 → 順流
        #   ・渦域が2車以上 → 渦
        #   ・逆流域が2車以上 → 逆流
        #   ・3車がそれぞれ別流域、または判定不能 → 逆流
        #   ※H主導寄せより後で適用し、主導ラインだけで順流へ戻りすぎるのを防ぐ。
        # =====================================================
        try:
            if not is_girls_like:
                def _zone_for_car_from_current_groups(_car_no):
                    try:
                        _car_no = int(_car_no)
                        if isinstance(line_def, dict):
                            for _gid, _members in line_def.items():
                                _members_i = [int(x) for x in (_members or []) if str(x).isdigit()]
                                if _car_no in _members_i:
                                    return _current_zone_for_line(_members_i)
                    except Exception:
                        pass
                    return "その他"

                _ko_top3 = []
                try:
                    _ko_top3 = [
                        int(c) for c, _ in sorted(
                            [(int(c), float(score_map.get(int(c), 0.0))) for c in score_map.keys()],
                            key=lambda x: (-x[1], x[0])
                        )[:3]
                    ]
                except Exception:
                    _ko_top3 = []

                if len(_ko_top3) >= 3:
                    _zone_counts = {"順流": 0, "渦": 0, "逆流": 0}
                    _zone_detail = []

                    for _c in _ko_top3:
                        _z = _zone_for_car_from_current_groups(_c)
                        if _z in _zone_counts:
                            _zone_counts[_z] += 1
                        else:
                            _z = "その他"
                        _zone_detail.append(f"{_c}:{_z}")

                    _ko_style = None
                    if _zone_counts.get("順流", 0) >= 2:
                        _ko_style = "順流"
                    elif _zone_counts.get("渦", 0) >= 2:
                        _ko_style = "渦"
                    elif _zone_counts.get("逆流", 0) >= 2:
                        _ko_style = "逆流"
                    else:
                        _ko_style = "逆流"

                    if _ko_style and _ko_style != recommend_style:
                        recommend_reason.append(
                            "KO上位3車流域多数決により"
                            f"{_ko_style}寄せ（" + "／".join(_zone_detail) + "）"
                        )
                        recommend_style = _ko_style
                        # KO上位3車の偏りは買い目に直結するため、最低B扱いにする
                        if confidence == "C":
                            confidence = "B"
                    elif _ko_style:
                        recommend_reason.append(
                            "KO上位3車流域多数決="
                            f"{_ko_style}（" + "／".join(_zone_detail) + "）"
                        )
        except Exception:
            pass

                # =====================================================
        # ガールズ補正
        #   ガールズはライン戦ではないため、
        #   無印押上げだけで渦に寄せすぎない
        # =====================================================
        try:
            if is_girls_like and recommend_style == "渦":
                recommend_style = "順流"
                recommend_reason.append("ガールズ/アドバンスのため渦寄せを順流扱いに補正")
        except Exception:
            pass

                # =====================================================
        # 信頼度の最終補正：展開評価・順当度・上位差を統合
        # =====================================================
        try:
            compact_label = str(globals().get("race_compact_label", ""))
            compact_gap = globals().get("race_compact_gap", None)

            def _down_conf(conf):
                if conf == "A":
                    return "B"
                if conf == "B":
                    return "C"
                return "C"

            conf_down_reasons = []

            # 波乱気味＋上位差小は、信頼度を1段階下げる
            if "波乱気味" in compact_label and compact_gap is not None:
                if float(compact_gap) < 1.0:
                    old_conf = confidence
                    confidence = _down_conf(confidence)
                    if confidence != old_conf:
                        conf_down_reasons.append(
                            f"波乱気味＋上位差小={float(compact_gap):.2f}"
                        )

            # 混戦＋波乱気味はB以上を出しすぎない
            if "混戦" in tenkai_txt and "波乱気味" in compact_label:
                if confidence in ("A", "B"):
                    old_conf = confidence
                    confidence = "C"
                    if confidence != old_conf:
                        conf_down_reasons.append("混戦＋波乱気味")

            # レースFRが不利域なら、AはBへ落とす
            if raceFR >= 0.65 and confidence == "A":
                confidence = "B"
                conf_down_reasons.append(f"レースFR不利域={raceFR:.3f}")

            # ライン偏差大なら、B以上を1段階下げる
            if sd >= 0.60:
                old_conf = confidence
                confidence = _down_conf(confidence)
                if confidence != old_conf:
                    conf_down_reasons.append("ライン偏差大")

            if conf_down_reasons:
                recommend_reason.append(
                    "信頼度補正：" + "／".join(conf_down_reasons)
                )

        except Exception:
            pass

               # =====================================================
        # 推奨戦法を＜短評＞の上に表示
        # =====================================================
        recommend_lines = []
        recommend_lines.append(
            f"推奨戦法：{recommend_style}"
        )

        # =====================================================
        # 買い間違い防止：推奨戦法の着順予想だけを強調表示
        #   目視用：7 → 1 → 5 ...
        #   コピー用：7152364
        # =====================================================
        try:
            _style_seq_map_for_display = globals().get("STYLE_SEQ_MAP", {})
            _recommended_seq = _style_seq_map_for_display.get(recommend_style, [])

            if not _recommended_seq:
                # 保険：STYLE_SEQ_MAPが空の場合は、直前で作った各戦法順から拾う
                _fallback_map = {
                    "順流": [int(x) for x in (out_j or []) if str(x).isdigit()],
                    "渦":   [int(x) for x in (out_v or []) if str(x).isdigit()],
                    "逆流": [int(x) for x in (out_u or []) if str(x).isdigit()],
                }
                _recommended_seq = _fallback_map.get(recommend_style, [])

            if _recommended_seq:
                _display_seq = " → ".join(str(int(x)) for x in _recommended_seq if str(x).isdigit())
                _copy_seq = "".join(str(int(x)) for x in _recommended_seq if str(x).isdigit())

                recommend_lines.append("")
                recommend_lines.append(f"✅ 推奨戦法：{recommend_style}")
                recommend_lines.append("")
                recommend_lines.append(f"【{recommend_style}メイン着順予想】")
                recommend_lines.append(_display_seq)
                recommend_lines.append("")
                recommend_lines.append(f"コピー用：{_copy_seq}")
                recommend_lines.append("")

                # st.markdown表示時に強調しやすいよう、HTML版も保持しておく
                globals()["RECOMMENDED_STYLE"] = recommend_style
                globals()["RECOMMENDED_STYLE_SEQ"] = _recommended_seq
                globals()["RECOMMENDED_STYLE_COPY"] = _copy_seq
        except Exception as _e:
            recommend_lines.append(f"推奨戦法表示生成不可（{_e}）")
            recommend_lines.append("")

        # =====================================================
        # 2車複 評価1軸候補
        # 推奨戦法の評価順だけを使う
        # =====================================================
        try:
            import math

            def _axis_rank(p):
                if p >= 0.40:
                    return "A", "買い候補強"
                if p >= 0.30:
                    return "B", "買い候補"
                if p >= 0.20:
                    return "C", "オッズ条件付き"
                if p >= 0.10:
                    return "D", "高配当条件"
                return "E", "ケン寄り"

            def _safe_odds_from_prob(p):
                if p <= 1e-12:
                    return None
                return 1.0 / float(p)

            def _style_axis_pairs(seq, score_map):
                """
                評価順seqの1位を軸にした2車複候補を作る。
                推定率はPlackett-Luce型の上位2着内ペア近似。
                """
                xs = []
                seen = set()

                for x in (seq or []):
                    if str(x).isdigit():
                        c = int(x)
                        if c not in seen:
                            seen.add(c)
                            xs.append(c)

                if len(xs) < 2:
                    return None, [], 0.0

                vals = []
                for c in xs:
                    vals.append(float(score_map.get(int(c), 0.0) or 0.0))

                mu = sum(vals) / max(len(vals), 1)
                var = sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)
                sdv = var ** 0.5
                if sdv <= 1e-9:
                    sdv = 1.0

                # 温度：小さいほどスコア差を強く見る
                temp = 1.65

                weights = {}
                for c, v in zip(xs, vals):
                    z = (v - mu) / (sdv * temp)
                    z = max(-6.0, min(6.0, z))
                    weights[int(c)] = math.exp(z)

                total_w = sum(weights.values())
                if total_w <= 1e-12:
                    return xs[0], [], 0.0

                axis = xs[0]
                wa = float(weights.get(axis, 0.0))

                pair_rows = []
                for opp in xs[1:]:
                    wb = float(weights.get(int(opp), 0.0))

                    # 無順序2車複：P(axis→opp) + P(opp→axis)
                    p1 = (wa / total_w) * (wb / max(total_w - wa, 1e-12))
                    p2 = (wb / total_w) * (wa / max(total_w - wb, 1e-12))
                    p_pair = max(0.0, p1 + p2)

                    pair_rows.append((axis, int(opp), p_pair))

                # 評価1軸の想定2着内率
                axis_rate = sum(p for _, _, p in pair_rows)

                return axis, pair_rows, axis_rate

            style_seq_map = globals().get("STYLE_SEQ_MAP", {})

            # 推奨戦法に応じた評価順を採用
            selected_style = recommend_style
            selected_seq = style_seq_map.get(selected_style, [])

            # 保険：推奨戦法のseqが空なら順流を使う
            if not selected_seq:
                selected_style = "順流"
                selected_seq = style_seq_map.get("順流", [])

            axis, pair_rows, axis_rate = _style_axis_pairs(selected_seq, score_map)
            axis_rank, axis_label = _axis_rank(axis_rate)

            # 他戦法で軸率が高いものを参考表示する
            ref_msgs = []
            selected_fr = _style_fr_for_recommend(selected_style)

            for other_style in ["順流", "渦", "逆流"]:
                if other_style == selected_style:
                    continue

                other_seq = style_seq_map.get(other_style, [])
                other_axis, _, other_rate = _style_axis_pairs(other_seq, score_map)

                if other_axis is None:
                    continue

                                # 推奨戦法より3%以上高い場合だけ参考表示
                if other_rate >= axis_rate + 0.03:
                    other_fr = _style_fr_for_recommend(other_style)

                    if other_fr < selected_fr - 0.05:
                        ref_msgs.append(
                            f"{other_style}評価1位の{int(other_axis)}は軸率高め。ただし{other_style}FR低めのため参考扱い。"
                        )
                    else:
                        ref_msgs.append(
                            f"{other_style}評価1位の{int(other_axis)}も軸候補。{other_style}警戒。"
                        )

            if axis is not None and pair_rows:
                recommend_lines.append(
                    f"軸評価：{axis_rank}［{axis_label}］"
                    f"（軸想定2着内率 {axis_rate*100:.0f}%）"
                )
                globals()["AXIS_EVAL_TOP_LINE"] = (
                    f"軸評価：{axis_rank}［{axis_label}］"
                    f"（軸想定2着内率 {axis_rate*100:.0f}%）"
                )

                try:
                    _compact_label_for_buy = str(
                        globals().get("race_compact_label", "未判定")
                    )
                    _compact_gap_for_buy = globals().get("race_compact_gap", None)

                    if _compact_gap_for_buy is not None:
                        recommend_lines.append(
                            f"順当度：{_compact_label_for_buy}［上位差={float(_compact_gap_for_buy):.2f}］"
                        )
                    else:
                        recommend_lines.append(
                            f"順当度：{_compact_label_for_buy}"
                        )

                except Exception:
                    pass

                recommend_lines.append("")

                recommend_lines.append("【2車複 評価軸候補】")
                recommend_lines.append(f"基準：{selected_style}メイン")
                recommend_lines.append("2車複想定軸：評価1・評価2")

                # =====================================================
                # 2車複候補一覧＋絞り推奨買目
                # 評価1・評価2を軸にする
                # 候補一覧：重複を削らない
                # 絞り推奨：推定率10%以上、かつ重複削除
                # =====================================================
                SHIBORI_MIN_PROB = 0.10
                shibori_items = []
                shibori_seen = set()

                def _format_nifuku_line(a, b, p):
                    odds = _safe_odds_from_prob(p)
                    if odds is None:
                        return f"{int(a)}-{int(b)}　推定率 0.0% ／ 足切り —"
                    return f"{int(a)}-{int(b)}　推定率 {p*100:.1f}% ／ 足切り {odds:.1f}倍以上"

                def _make_axis_pair_rows(seq, score_map, axis_index=0):
                    """
                    評価順seqの axis_index 番目を軸にした2車複候補を作る。
                    axis_index=0 → 評価1軸
                    axis_index=1 → 評価2軸
                    """
                    xs = []
                    seen = set()

                    for x in (seq or []):
                        if str(x).isdigit():
                            c = int(x)
                            if c not in seen:
                                seen.add(c)
                                xs.append(c)

                    if len(xs) < 2 or axis_index >= len(xs):
                        return None, [], 0.0

                    vals = []
                    for c in xs:
                        vals.append(float(score_map.get(int(c), 0.0) or 0.0))

                    mu = sum(vals) / max(len(vals), 1)
                    var = sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)
                    sdv = var ** 0.5
                    if sdv <= 1e-9:
                        sdv = 1.0

                    temp = 1.65

                    weights = {}
                    for c, v in zip(xs, vals):
                        z = (v - mu) / (sdv * temp)
                        z = max(-6.0, min(6.0, z))
                        weights[int(c)] = math.exp(z)

                    total_w = sum(weights.values())
                    if total_w <= 1e-12:
                        return xs[axis_index], [], 0.0

                    axis2 = int(xs[axis_index])
                    wa = float(weights.get(axis2, 0.0))

                    rows = []
                    for opp in xs:
                        opp = int(opp)
                        if opp == axis2:
                            continue

                        wb = float(weights.get(opp, 0.0))

                        # 無順序2車複：P(axis→opp) + P(opp→axis)
                        p1 = (wa / total_w) * (wb / max(total_w - wa, 1e-12))
                        p2 = (wb / total_w) * (wa / max(total_w - wb, 1e-12))
                        p_pair = max(0.0, p1 + p2)

                        rows.append((axis2, opp, p_pair))

                    rate = sum(p for _, _, p in rows)
                    return axis2, rows, rate

                for _axis_index, _label in [(0, "評価1軸"), (1, "評価2軸")]:
                    _axis_car, _rows, _rate = _make_axis_pair_rows(
                        selected_seq,
                        score_map,
                        axis_index=_axis_index
                    )

                    if _axis_car is None or not _rows:
                        continue

                    recommend_lines.append("")
                    recommend_lines.append(f"{_label}：{int(_axis_car)}")

                    _rows_sorted = sorted(
                        _rows,
                        key=lambda t: int(t[1])
                    )

                    for a, b, p in _rows_sorted:
                        line = _format_nifuku_line(a, b, p)
                        recommend_lines.append(line)

                        if float(p) >= SHIBORI_MIN_PROB:
                            # 絞り推奨だけは2車複なので重複削除
                            k = tuple(sorted((int(a), int(b))))
                            if k not in shibori_seen:
                                shibori_seen.add(k)
                                shibori_items.append((a, b, p, line))

                if ref_msgs:
                    recommend_lines.append("参考：" + "／".join(ref_msgs))

                # 絞り推奨買目を別枠で表示
                if shibori_items:
                    recommend_lines.append("")
                    recommend_lines.append("**【絞り推奨買目】（推定率10％以上が基準／重複削除）**")

                    for a, b, p, line in shibori_items:
                        recommend_lines.append(f"**{line}**")

                                # =====================================================
                # 仮想単勝：2車単 軸→全
                # 競輪には単勝がないため、2車単「軸→全」を仮想単勝として扱う
                # 評価1軸・評価2軸を表示
                # =====================================================
                try:
                    def _axis_win_prob(seq, score_map, axis_car):
                        xs = []
                        seen = set()

                        for x in (seq or []):
                            if str(x).isdigit():
                                c = int(x)
                                if c not in seen:
                                    seen.add(c)
                                    xs.append(c)

                        if not xs or int(axis_car) not in xs:
                            return 0.0

                        vals = []
                        for c in xs:
                            vals.append(float(score_map.get(int(c), 0.0) or 0.0))

                        mu = sum(vals) / max(len(vals), 1)
                        var = sum((v - mu) ** 2 for v in vals) / max(len(vals), 1)
                        sdv = var ** 0.5
                        if sdv <= 1e-9:
                            sdv = 1.0

                        # 2車複推定率と同じ温度を使用
                        temp = 1.65

                        weights = {}
                        for c, v in zip(xs, vals):
                            z = (v - mu) / (sdv * temp)
                            z = max(-6.0, min(6.0, z))
                            weights[int(c)] = math.exp(z)

                        total_w = sum(weights.values())
                        if total_w <= 1e-12:
                            return 0.0

                        return float(weights.get(int(axis_car), 0.0)) / total_w

                    def _unique_seq(seq):
                        xs = []
                        seen = set()
                        for x in (seq or []):
                            if str(x).isdigit():
                                c = int(x)
                                if c not in seen:
                                    seen.add(c)
                                    xs.append(c)
                        return xs

                    _seq_unique = _unique_seq(selected_seq)

                    recommend_lines.append("")
                    recommend_lines.append("【仮想単勝：2車単 軸→全】")

                    for _axis_index, _label in [(0, "評価1軸"), (1, "評価2軸")]:
                        if _axis_index >= len(_seq_unique):
                            continue

                        _axis_car = int(_seq_unique[_axis_index])
                        axis_win_rate = _axis_win_prob(
                            selected_seq,
                            score_map,
                            _axis_car
                        )

                        if axis_win_rate <= 1e-12:
                            continue

                        theoretical_odds = 1.0 / axis_win_rate

                        # 軸→全の点数。7車なら6点、5車なら4点、9車なら8点。
                        n_tansho_points = max(len(_seq_unique) - 1, 1)

                        # 2車単「軸→全」は、最安目ではなく合成オッズで見る
                        required_composite_odds = theoretical_odds

                        # 実戦では推定誤差を考えて少し上乗せ
                        practical_composite_odds = theoretical_odds * 1.10

                        recommend_lines.append("")
                        recommend_lines.append(f"{_label}：{int(_axis_car)}")
                        recommend_lines.append(
                            f"軸1着推定率：{axis_win_rate*100:.1f}%"
                        )
                        
                        recommend_lines.append(
                            f"2車単 軸→全 必要合成オッズ：{required_composite_odds:.1f}倍以上"
                        )
                        recommend_lines.append(
                            f"実戦目安：合成{practical_composite_odds:.1f}倍以上なら検討"
                        )
                        recommend_lines.append(
                            f"参考：均等買いのトリガミ回避は各目{float(n_tansho_points):.1f}倍以上"
                        )
                except Exception:
                    pass

                recommend_lines.append("")

        except Exception as _e:
            recommend_lines.append(
                f"【2車複 評価1軸候補】生成不可（{_e}）"
            )
            recommend_lines.append("")

        # 推奨理由は短評内に残す
        lines_out.append(
            f"・推奨理由：{'／'.join(recommend_reason)}"
        )

    except Exception as _e:
        recommend_lines = []
        recommend_lines.append(
            f"推奨戦法：判定不可（{_e}）"
        )
        recommend_lines.append("")

    # =====================================================
    # 冒頭表示用：展開評価の直後に軸評価を1行だけ差し込む
    # =====================================================
    try:
        _axis_top_line = globals().get("AXIS_EVAL_TOP_LINE", "")

        if _axis_top_line:
            for _i, _s in enumerate(note_sections):
                if str(_s).startswith("展開評価："):
                    if (
                        _i + 1 >= len(note_sections)
                        or str(note_sections[_i + 1]) != _axis_top_line
                    ):
                        note_sections.insert(_i + 1, _axis_top_line)
                    break

    except Exception:
        pass

    note_sections.extend(recommend_lines)
    note_sections.extend(lines_out)
    note_sections.append("")
    globals()["note_sections"] = note_sections

    globals()["note_sections"] = note_sections

except Exception as _e:
    try:
        ns = globals().get("note_sections", None)
        if not isinstance(ns, list):
            ns = []
            globals()["note_sections"] = ns

        ns.append("")
        ns.append("＜短評＞")
        ns.append(f"・出力生成中に例外が発生しました: {_e}")
        ns.append("判定：混戦")

    except Exception:
        pass

# =========================
# note用コピーエリア：期待値軸＋実車番フォーメーション
# =========================

note_text = "\n".join(note_sections)

st.markdown("### 📋 note用（コピーエリア）")

# -----------------------------------------
# 期待値軸判定用：◎〇△× 車番入力
# ※公開コピーには、市場名・外部名は出さない
# ※入力された印は「当たりやすさ」ではなく、市場人気による期待値減衰として扱う
# -----------------------------------------
# 期待値軸判定用の市場印は、計算反映前に snapshot へ固定済み。
# ここでは再入力させず、反映済み値だけを使う。
market_honmei_raw = snapshot.get("market_honmei_raw", "—")
market_taikou_raw = snapshot.get("market_taikou_raw", "—")
market_tan_raw = snapshot.get("market_tan_raw", "—")
market_batsu_raw = snapshot.get("market_batsu_raw", "—")

def _to_car_int_or_none(v):
    try:
        s = str(v).strip()
        if not s or s == "—":
            return None
        x = int(s)
        return x if 1 <= x <= 9 else None
    except Exception:
        return None


market_honmei = _to_car_int_or_none(market_honmei_raw)
market_taikou = _to_car_int_or_none(market_taikou_raw)
market_tan = _to_car_int_or_none(market_tan_raw)
market_batsu = _to_car_int_or_none(market_batsu_raw)

# 車番→印。
# v20: 原則として、入力画面の「車番ごとの外部印」をそのまま使う。
# 以前は ◎/〇/△/× から車番へ圧縮した値だけで復元していたため、
# 三連複の評価重複欄で「車番1が無印なのに◎/〇になる」などのズレが起き得た。
_market_mark_snapshot = snapshot.get("market_mark_by_car", {})
_VALID_MARKS_LOCAL = {"◎", "〇", "○", "△", "▲", "×"}

def _normalize_market_mark_local(_mark):
    _mk = str(_mark or "").strip()
    if _mk == "○":
        return "〇"
    if _mk == "▲":
        return "△"
    return _mk

market_mark_map = {}
# v47: market_mark_by_car は全車ぶん「—」を持つことがある。
# その場合に「dictがあるからfallbackしない」と、市場印が空扱いになって妙味ptが10のままになる。
# 先に車番ごとの有効印を拾い、その後で旧raw値も必ず補完する。
if isinstance(_market_mark_snapshot, dict) and _market_mark_snapshot:
    for _car, _mark in _market_mark_snapshot.items():
        try:
            _ci = int(_car)
        except Exception:
            continue
        _mk = _normalize_market_mark_local(_mark)
        if _mk in _VALID_MARKS_LOCAL:
            market_mark_map[_ci] = _mk

# v190:
# 反映後の計算では snapshot に保存した市場印だけを使う。
# ここで st.session_state の現在値を再取得すると、
# 「反映ボタンを押した固定値」と「画面上の未反映値」が混ざり、
# 市場印が空扱い/旧R扱いになって妙味ptが10.0に張り付く原因になる。

# 旧snapshot用・または market_mark_by_car が「—」だけだった時の補完。
# ここは setdefault ではなく、有効な旧rawがある場合は上書きする。
# ただし上の車番別radioが有効なら同じ内容になる。
for _car, _mark in [
    (market_honmei, "◎"),
    (market_taikou, "〇"),
    (market_tan, "△"),
    (market_batsu, "×"),
]:
    if _car is None:
        continue
    try:
        market_mark_map[int(_car)] = _mark
    except Exception:
        pass


def _uniq_keep(seq):
    out = []
    seen = set()
    for x in seq:
        try:
            xi = int(x)
        except Exception:
            continue
        if xi not in seen:
            out.append(xi)
            seen.add(xi)
    return out


def _fmt_cars(seq):
    return "".join(str(int(x)) for x in seq if str(x).isdigit())


def _count_nishatan(col1, col2):
    return sum(1 for a in col1 for b in col2 if int(a) != int(b))


def _count_sanrentan(col1, col2, col3):
    return sum(
        1
        for a in col1
        for b in col2
        for c in col3
        if len({int(a), int(b), int(c)}) == 3
    )


def _count_sanpuku(col1, col2, col3):
    combos = set()
    for a in col1:
        for b in col2:
            for c in col3:
                s = tuple(sorted([int(a), int(b), int(c)]))
                if len(set(s)) == 3:
                    combos.add(s)
    return len(combos)


def _find_line_members_of_car(line_def_obj, car):
    try:
        car = int(car)
        if isinstance(line_def_obj, dict):
            for _, mem in line_def_obj.items():
                mm = [int(x) for x in (mem or []) if str(x).isdigit()]
                if car in mm:
                    return mm
    except Exception:
        pass
    return []


def _find_line_members_of_car_from_note_text(note_text_obj, car):
    """
    line_def が globals に無い/取れない場合の保険。
    note本文の「ライン　73　16　524」から評価1の所属ラインを復元する。
    """
    try:
        car = int(car)
        txt = str(note_text_obj or "")
        m = re.search(r"^ライン\s+(.+)$", txt, flags=re.MULTILINE)
        if not m:
            return []
        part = m.group(1).strip()
        # 全角スペース・半角スペースで分割。数字以外は落とす。
        chunks = re.split(r"[\s　]+", part)
        for ch in chunks:
            nums = [int(x) for x in re.findall(r"\d", ch)]
            if car in nums:
                return nums
    except Exception:
        pass
    return []



def _parse_line_members_from_note_text(note_text_obj):
    """
    note本文の「ライン　416　27　3　5」から、ライン配列を復元する。
    返り値例：[[4,1,6], [2,7], [3], [5]]
    """
    try:
        txt = str(note_text_obj or "")
        m = re.search(r"^ライン\s+(.+)$", txt, flags=re.MULTILINE)
        if not m:
            return []
        part = m.group(1).strip()
        chunks = re.split(r"[\s　]+", part)
        out = []
        for ch in chunks:
            nums = []
            for x in re.findall(r"\d", ch):
                xi = int(x)
                if 1 <= xi <= 9 and xi not in nums:
                    nums.append(xi)
            if nums:
                out.append(nums)
        return out
    except Exception:
        return []


def _line_members_list_from_line_def(line_def_obj):
    """
    globals の line_def からライン配列を作る保険。
    """
    try:
        if isinstance(line_def_obj, dict):
            out = []
            for _, mem in line_def_obj.items():
                nums = []
                for x in (mem or []):
                    if str(x).isdigit():
                        xi = int(x)
                        if 1 <= xi <= 9 and xi not in nums:
                            nums.append(xi)
                if nums:
                    out.append(nums)
            return out
    except Exception:
        pass
    return []


def _rank_lines_by_order(line_members_list, order_seq):
    """
    各ラインを、推奨順/KO順で一番早く出る車を代表順位として並べる。
    評価順そのものではなく、ライン単位で列へ割り振るための基礎。
    """
    order = [int(x) for x in (order_seq or []) if str(x).isdigit()]
    pos = {car: i for i, car in enumerate(order)}

    def key(mem):
        best = min([pos.get(int(c), 999) for c in mem] or [999])
        # 同順位の保険として、ライン先頭の推奨位置も見る
        head_pos = pos.get(int(mem[0]), 999) if mem else 999
        return (best, head_pos, len(mem))

    return sorted([list(map(int, mem)) for mem in (line_members_list or []) if mem], key=key)



def _pick_eval1_line_promote_car(eval1_line_members, current_col2, mark_map):
    """
    評価1ライン内の印付き未採用車を、2列目へ1車だけ繰り上げる。

    目的：
    ・評価1を頭に置くなら、評価1ライン内の番手/後続が2着に残る筋を拾う。
    ・ただし点数増を避けるため、繰り上げは1車だけ。
    ・◎〇△×の順で優先し、同格ならライン順（番手優先）にする。
    """
    try:
        line = [int(x) for x in (eval1_line_members or []) if str(x).isdigit()]
        already = {int(x) for x in (current_col2 or []) if str(x).isdigit()}
        mark_map = {int(k): str(v) for k, v in (mark_map or {}).items()}
        mark_rank = {"◎": 4, "〇": 3, "△": 2, "×": 1}

        candidates = []
        for idx, car in enumerate(line):
            if int(car) in already:
                continue
            mk = mark_map.get(int(car), "無印")
            if mk not in mark_rank:
                continue
            candidates.append((mark_rank[mk], -idx, int(car)))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        return int(candidates[0][2])

    except Exception:
        return None


def _calc_expect_axis_score_label(col1_cars, col2_cars, role1, mark_map):
    """
    期待値軸を点数化する。

    基本思想：
    ・信頼度ではなく、市場印とのズレによる配当妙味を見る。
    ・2車単フォメを基準に、1列目候補と2列目専用候補を分けて評価する。
    ・1列目に市場印が付くほど人気寄りで期待値は下がりやすい。
    ・2列目だけの市場印は、相手人気として軽く減点する。
    ・評価1が無印なら、市場からズレた期待値妙味として加点する。

    期待値点 = 10
      - 1列目印減点
      - 2列目専用印減点
      + 評価1印補正
    """
    try:
        col1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
        col2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
        r1 = int(role1)
        mark_map = {int(k): str(v) for k, v in (mark_map or {}).items()}

        head_penalty = {"◎": 4.0, "〇": 3.0, "△": 1.5, "×": 0.75, "無印": 0.0}
        tail_penalty = {"◎": 2.0, "〇": 1.5, "△": 0.75, "×": 0.40, "無印": 0.0}

        score = 10.0
        role_marks = []

        # 1列目候補は頭として市場に売れやすいため強めに減点
        for car in col1:
            mk = mark_map.get(int(car), "無印")
            role_marks.append(mk)
            score -= head_penalty.get(mk, 0.0)

        # 2列目だけの候補は相手人気なので軽めに減点
        col1_set = set(col1)
        for car in col2:
            if int(car) in col1_set:
                continue
            mk = mark_map.get(int(car), "無印")
            role_marks.append(mk)
            score -= tail_penalty.get(mk, 0.0)

        # 評価1の印による補正。信頼度ではなく、市場とのズレを表す。
        r1_mark = mark_map.get(r1, "無印")
        r1_bonus_map = {"無印": 1.0, "×": 0.5, "△": 0.0, "〇": -0.5, "◎": -1.0}
        score += r1_bonus_map.get(r1_mark, 0.0)

        score = max(0.0, min(10.0, float(score)))

        # 表示ランクだけを調整。
        # 6.6点のような「期待値はあるが荒れ寄り」の形をAAに上げすぎない。
        # 8.0以上はズレすぎの荒領域として扱う。
        if score >= 8.0:
            label = "荒"
        elif score >= 6.8:
            label = "AA"
        elif score >= 5.5:
            label = "A"
        elif score >= 4.5:
            label = "B"
        elif score >= 3.5:
            label = "C"
        else:
            label = "低"

        return label, round(score, 1), role_marks

    except Exception:
        return "C", None, []



def _myoumi_mark_penalty(mark: str, role: str) -> float:
    """
    妙味ピックアップ用の印減点。
    思想：
    ・外部印とVeloBi買い目構造が被るほど、市場評価と一致して妙味は下がる。
    ・1列目の印被りは強く減点、2列目は中、3列目は軽く減点。
    ・無印は減点しない。×は軽い減点に留める。
    """
    mark = str(mark or "無印")
    if role == "head":
        return {"◎": 4.0, "〇": 3.0, "△": 1.5, "×": 0.75, "無印": 0.0}.get(mark, 0.0)
    if role == "tail":
        return {"◎": 2.0, "〇": 1.5, "△": 0.75, "×": 0.40, "無印": 0.0}.get(mark, 0.0)
    if role == "third":
        return {"◎": 1.0, "〇": 0.75, "△": 0.40, "×": 0.20, "無印": 0.0}.get(mark, 0.0)
    return 0.0


def _myoumi_eval1_bonus(car: int, role1: int, mark_map: dict) -> float:
    """
    評価1を1列目に置く場合だけ、市場ズレを内部基準に反映する。
    評価1が無印・×なら妙味を上げ、◎なら市場評価と一致しているので下げる。
    """
    try:
        if int(car) != int(role1):
            return 0.0
        mk = {int(k): str(v) for k, v in (mark_map or {}).items()}.get(int(role1), "無印")
        return {"無印": 1.0, "×": 0.5, "△": 0.0, "〇": -0.5, "◎": -1.0}.get(mk, 0.0)
    except Exception:
        return 0.0


def _myoumi_market_pair_penalty(marks) -> float:
    """
    2車系用の本線ペア追加減点。
    ◎〇のように市場評価ど真ん中の組み合わせは、的中候補ではあっても妙味候補ではないため強く落とす。
    """
    ms = {str(x or "無印") for x in marks}
    if "◎" in ms and "〇" in ms:
        return 1.5
    if "◎" in ms and "△" in ms:
        return 0.8
    if "〇" in ms and "△" in ms:
        return 0.6
    if "◎" in ms and "×" in ms:
        return 0.3
    if "〇" in ms and "×" in ms:
        return 0.2
    return 0.0


def _has_valid_market_marks_for_myoumi(mark_map: dict) -> bool:
    """市場印が1つ以上反映されているか。未反映時に妙味10.0張り付きを防ぐための保険。"""
    try:
        valid = {"◎", "〇", "○", "△", "▲", "×"}
        for v in (mark_map or {}).values():
            mk = str(v or "").strip()
            if mk in valid:
                return True
        return False
    except Exception:
        return False

def _resolve_market_mark_for_car_myoumi(car: int, mark_map: dict) -> str:
    """
    妙味計算専用の市場印取得。

    v190:
    反映済み snapshot から作った mark_map だけを信用する。
    st.session_state をここで再検索すると、未反映の画面値や別Rのradio値を拾い、
    反映ボタンの固定計算とズレるため使わない。
    """
    try:
        c = int(car)
    except Exception:
        return "無印"

    valid = {"◎", "〇", "○", "△", "▲", "×"}

    def norm(v):
        mk = str(v or "").strip()
        if mk == "○":
            mk = "〇"
        if mk == "▲":
            mk = "△"
        if mk in valid:
            return mk
        return "無印"

    try:
        mm = {int(k): norm(v) for k, v in (mark_map or {}).items()}
        return mm.get(c, "無印")
    except Exception:
        return "無印"


def _myoumi_market_trio_penalty(marks) -> float:
    """
    3連系用の本線構成追加減点。
    ◎〇△が同居するほど、市場本線寄りとして妙味を下げる。
    """
    ms = {str(x or "無印") for x in marks}
    pen = 0.0

    # ◎〇同居は強い市場本線扱い
    if "◎" in ms and "〇" in ms:
        pen += 1.0

    # ◎〇△が揃う場合はさらに本線寄り
    if {"◎", "〇", "△"}.issubset(ms):
        pen += 1.0
    elif "◎" in ms and "△" in ms:
        pen += 0.4
    elif "〇" in ms and "△" in ms:
        pen += 0.3

    # 印付き3車で固まりすぎる場合は軽く追加減点
    marked_count = sum(1 for m in marks if str(m or "無印") in {"◎", "〇", "△", "×"})
    if marked_count >= 3:
        pen += 0.3

    return pen


def _myoumi_score_2kei(a: int, b: int, role1: int, mark_map: dict) -> float:
    """
    2車系ピックアップ用。
    a-b の順番はフォメ列順を保持する。
    実オッズではなく、外部印との被りから見た内部妙味pt。

    v50方針：
    ・2車複は「軸の印」より「相手の印の薄さ」を重視する。
      軸が◎でも、相手が無印なら妙味は残す。
    ・ただし、相手が△/〇/◎なら市場にも拾われているので明確に下げる。
    ・市場印を mark_map だけに頼らず、session_state からも再取得する。
    """
    mm = {int(k): str(v) for k, v in (mark_map or {}).items()}

    # v190:
    # 市場印が1つも反映されていない状態を「全車無印＝全買い目が超妙味」と誤解しない。
    # 反映失敗・未入力時は中立値に落とし、A++張り付きを防ぐ。
    if not _has_valid_market_marks_for_myoumi(mm):
        return 7.0

    ma = _resolve_market_mark_for_car_myoumi(int(a), mm)
    mb = _resolve_market_mark_for_car_myoumi(int(b), mm)

    # 2車複では、軸が市場◎でも「相手が売れていない」なら妙味は残る。
    # そのため head 減点は軽め、tail 減点は強めにする。
    head_penalty = {"◎": 0.8, "〇": 0.55, "△": 0.30, "×": 0.15, "無印": 0.0}
    tail_penalty = {"◎": 2.4, "〇": 1.7, "△": 1.2, "×": 0.45, "無印": 0.0}

    # 相手側の市場軽視を妙味として見る。
    tail_bonus = {"無印": 1.0, "×": 0.55, "△": 0.00, "〇": -0.35, "◎": -0.70}

    score = 9.2
    score -= head_penalty.get(ma, 0.0)
    score -= tail_penalty.get(mb, 0.0)
    score += tail_bonus.get(mb, 0.0)

    # ◎×△、◎×〇などは市場にも相手が見えているので追加で落とす。
    score -= _myoumi_market_pair_penalty([ma, mb]) * 0.75

    # 評価1が市場無印なら少しだけ上げ、◎なら少しだけ下げる。
    # 軸印だけで妙味を殺さないため係数は小さくする。
    score += 0.30 * _myoumi_eval1_bonus(int(a), int(role1), mm)

    return round(max(0.0, min(10.0, score)), 1)


def _myoumi_velobi_rank_penalty_3kei(a: int, b: int, c: int, rec_order_for_forme=None) -> float:
    """
    三連複妙味pt用のVeloBi順位補正。

    v44方針：
    ・外部印/WIN側の低評価だけで10.0ptへ張り付かないよう、VeloBi順流順位を強めに反映する。
    ・2列目で薄い車を使う方を重く減点し、3列目の穴は少しだけ許容する。
    ・妙味を消すのではなく、10点横並びを崩して買い目順位として使える差を付ける。
    """
    try:
        order = [int(x) for x in (rec_order_for_forme or []) if str(x).strip().isdigit()]
        if not order:
            return 0.0
        rank = {car: i + 1 for i, car in enumerate(order)}
        n = max(len(order), 1)

        def one_pen(car: int, role: str) -> float:
            r = int(rank.get(int(car), n))
            # v43はここが緩すぎて、WIN側ズレが優先されすぎた。
            if r <= 2:
                base = 0.0
            elif r == 3:
                base = 0.15
            elif r == 4:
                base = 0.55
            elif r == 5:
                base = 1.05
            elif r == 6:
                base = 1.55
            else:
                base = 2.05

            if role == "head":
                base *= 0.45
            elif role == "third":
                base *= 0.80
            else:  # tail / 2列目
                base *= 1.10
            return base

        ra = rank.get(int(a), n)
        rb = rank.get(int(b), n)
        rc = rank.get(int(c), n)
        pen = one_pen(int(a), "head") + one_pen(int(b), "tail") + one_pen(int(c), "third")

        avg_rank = (ra + rb + rc) / 3.0
        if avg_rank >= 5.0:
            pen += 0.85
        elif avg_rank >= 4.0:
            pen += 0.45
        elif avg_rank >= 3.4:
            pen += 0.20

        # 6〜7位級を含む組み合わせは、外部印ズレだけで最上位にしない。
        if max(ra, rb, rc) >= 7:
            pen += 0.45
        elif max(ra, rb, rc) >= 6:
            pen += 0.25

        return float(max(0.0, min(4.2, pen)))
    except Exception:
        return 0.0


def _myoumi_score_3kei(a: int, b: int, c: int, role1: int, mark_map: dict, rec_order_for_forme=None) -> float:
    """
    3連系ピックアップ用。
    a-b-c の順番はフォメ列順を保持する。
    実オッズではなく、外部印との被りから見た内部妙味pt。

    v45方針：
    ・三連複妙味は「外部印が軽い＝即10点」ではなく、VeloBi本体と市場印の重なりを見る。
    ・特に1列目軸が市場印付きなら、妙味の上限をキャップする。
      例：軸が△なら、ズレはあるが市場にも拾われているので10点にはしない。
    """
    mm = {int(k): str(v) for k, v in (mark_map or {}).items()}
    ma = mm.get(int(a), "無印")
    mb = mm.get(int(b), "無印")
    mc = mm.get(int(c), "無印")

    # v49:
    # 三連複も強制キャップは撤廃。
    # ただし基礎点を少し高めに戻し、VeloBi順位減点で差を付ける。
    # 目的は「10点横並びを崩す」ことであって、「△軸だから妙味を消す」ことではない。
    score = 10.2
    score -= _myoumi_mark_penalty(ma, "head")
    score -= _myoumi_mark_penalty(mb, "tail")
    score -= _myoumi_mark_penalty(mc, "third")
    score -= _myoumi_market_trio_penalty([ma, mb, mc])
    score -= _myoumi_velobi_rank_penalty_3kei(int(a), int(b), int(c), rec_order_for_forme)
    score += 0.35 * _myoumi_eval1_bonus(int(a), int(role1), mm)

    return round(max(0.0, min(10.0, score)), 1)


# ==============================
# 妙味pt 通過基準
# ==============================
# 5.0pt基準では候補が広がりすぎるため、実戦買目の通過基準を引き上げる。
# 2車複は8.5pt以上、三連複は8.0pt以上を標準にする。
MYOUMI_PASS_THRESHOLD_2KEI = float(globals().get("MYOUMI_PASS_THRESHOLD_2KEI", 8.5))
MYOUMI_PASS_THRESHOLD_3KEI = float(globals().get("MYOUMI_PASS_THRESHOLD_3KEI", 8.0))
# ワイドは現時点では未採用。2車複と三連複だけで表示する。

# 評価重複枠：妙味通過ではないが、外部印とVeloBi評価上位が「同じ車で」重なる本線寄りの買目。
# ここは「当たりやすいが安い」確認枠。
# 2車複は2車とも「外部印あり＋順流評価1〜4」を満たす場合だけ採用する。
# 片方が無印、または片方が順流評価外なら評価重複には含めない。
EVAL_OVERLAP_MIN_2KEI = float(globals().get("EVAL_OVERLAP_MIN_2KEI", 5.0))
EVAL_OVERLAP_MAX_2KEI = int(globals().get("EVAL_OVERLAP_MAX_2KEI", 3))
# v19修正：三連複は表示順と印・順流順位の注記順を必ず一致させる。
# 評価重複三連複：1列目-2列目-3列目の中で外部印が重なる三連複。
# ここは妙味ではなく「評価がかぶる安い本線寄り」の確認枠なので、妙味ptでは切らない。
EVAL_OVERLAP_MIN_3KEI = float(globals().get("EVAL_OVERLAP_MIN_3KEI", 5.0))
EVAL_OVERLAP_MAX_3KEI = int(globals().get("EVAL_OVERLAP_MAX_3KEI", 3))
MARKED_SET = {"◎", "〇", "○", "△", "▲", "×"}


def _is_market_marked(car: int, mark_map: dict) -> bool:
    try:
        mk = {int(k): str(v) for k, v in (mark_map or {}).items()}.get(int(car), "無印")
        return mk in MARKED_SET
    except Exception:
        return False


def _market_mark_label(car: int, mark_map: dict) -> str:
    try:
        return {int(k): str(v) for k, v in (mark_map or {}).items()}.get(int(car), "無印")
    except Exception:
        return "無印"


def _rank_label_from_order(car: int, rec_order_for_forme=None) -> str:
    """順流メイン順の順位ラベルを返す。例：順流3位"""
    try:
        order = [int(x) for x in (rec_order_for_forme or []) if str(x).isdigit()]
        c = int(car)
        if c in order:
            return f"順流{order.index(c) + 1}位"
    except Exception:
        pass
    return ""


def _overlap_note_for_car(car: int, mark_map: dict, rec_order_for_forme=None, top_n: int = 4) -> str:
    """
    評価重複欄の1車分ラベル。
    外部印とVeloBi評価が両方ある場合は両方出す。
    無印は表示せず、順流順位などVeloBi側の根拠を出す。
    例：〇・順流1位 / 順流3位
    """
    mk = _market_mark_label(int(car), mark_map)
    rank = _rank_label_from_order(int(car), rec_order_for_forme)
    parts = []
    if str(mk) in MARKED_SET:
        parts.append(str(mk))
    try:
        order = [int(x) for x in (rec_order_for_forme or []) if str(x).isdigit()]
        if int(car) in order and order.index(int(car)) < int(top_n):
            parts.append(rank)
        elif rank and not parts:
            # top_n外でも、無印だけで終わらせないため順流順位を表示
            parts.append(rank)
    except Exception:
        if rank:
            parts.append(rank)
    if not parts:
        return "列評価"
    return "・".join(parts)


def _collect_eval_overlap_2kei(col1_cars, col2_cars, role1, mark_map, exclude_keys=None, rec_order_for_forme=None):
    """
    評価重複2車複を集める。
    条件：
      ・1列目-2列目の2車複候補
      ・妙味通過枠に既に出ていない
      ・2車とも、同じ車に「外部印＋順流評価1〜4」が重なる組み合わせ
      ・評価重複は的中率補助枠なので、妙味ptでは足切りしない

    位置づけ：
      妙味ではなく、的中率を支える安い本線確認枠。
      単なる外部印同士でもなく、単なる順流上位同士でもなく、印と順流評価が同じ車で重なるものだけを見る。
    """
    try:
        exclude_keys = set(exclude_keys or set())
        all_items = _all_2kei_point_items(col1_cars, col2_cars, role1, mark_map)
        rec_order = [int(x) for x in (rec_order_for_forme or []) if str(x).isdigit()]
        if rec_order:
            velobi_top4 = set(rec_order[:4])
        else:
            # 念のためのフォールバック：列評価の前方から4車
            tmp = []
            for x in list(col1_cars or []) + list(col2_cars or []):
                try:
                    xi = int(x)
                    if xi not in tmp:
                        tmp.append(xi)
                except Exception:
                    pass
            velobi_top4 = set(tmp[:4])
        out = []
        for sc, a, b in all_items:
            key = tuple(sorted((int(a), int(b))))
            if key in exclude_keys:
                continue
            # 評価重複は「印と順流評価の重なり」を見る枠。
            # 妙味ptは表示・並び順の参考に使うが、足切りには使わない。
            ma = _is_market_marked(a, mark_map)
            mb = _is_market_marked(b, mark_map)
            a_top = int(a) in velobi_top4
            b_top = int(b) in velobi_top4

            # 2車複の評価重複は、同じ車に「外部印＋順流評価1〜4」が重なることを条件にする。
            # したがって2車とも、外部印あり かつ 順流評価1〜4でなければ出さない。
            # 例：◎・順流1位 / 〇・順流3位 のような形だけを評価重複とする。
            if not (ma and mb and a_top and b_top):
                continue

            marked_count = int(ma) + int(mb)
            top_count = int(a_top) + int(b_top)
            out.append((marked_count, top_count, float(sc), int(a), int(b)))

        out.sort(key=lambda x: (-x[2], -x[0], -x[1], x[3], x[4]))
        return [(sc, a, b, marked_count, top_count) for marked_count, top_count, sc, a, b in out[:EVAL_OVERLAP_MAX_2KEI]]
    except Exception:
        return []



def _collect_eval_overlap_3kei(col1_cars, col2_cars, col3_cars, role1, mark_map, rec_order_for_forme=None):
    """
    評価重複三連複を集める。

    条件：
      ・1列目-2列目-3列目の三連複候補
      ・通常の上位123固定ではなく、列評価の構造を通す
      ・3車すべてで、外部印と順流評価1〜4が同じ車に重なる組み合わせだけを採用

    重要：
      評価重複三連複は「妙味ptが高い買目」ではない。
      外部印と順流評価が同じ車で重なっている、的中率補助の安い本線候補。
      そのため、妙味ptの通過基準では切らない。
      表示では、無印の代わりに順流順位などVeloBi側の根拠を出す。
    """
    try:
        c1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
        c2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
        c3 = [int(x) for x in (col3_cars or []) if str(x).isdigit()]
        if not c1 or not c2 or not c3:
            return []

        rec_order = [int(x) for x in (rec_order_for_forme or []) if str(x).isdigit()]
        if rec_order:
            velobi_top4 = set(rec_order[:4])
        else:
            tmp = []
            for x in c1 + c2 + c3:
                if int(x) not in tmp:
                    tmp.append(int(x))
            velobi_top4 = set(tmp[:4])

        out = []
        seen = set()

        for i, a in enumerate(c1):
            for j, b in enumerate(c2):
                for k, c in enumerate(c3):
                    a = int(a); b = int(b); c = int(c)
                    if len({a, b, c}) != 3:
                        continue

                    key = tuple(sorted((a, b, c)))
                    if key in seen:
                        continue
                    seen.add(key)

                    marks = [
                        _market_mark_label(a, mark_map),
                        _market_mark_label(b, mark_map),
                        _market_mark_label(c, mark_map),
                    ]
                    marked_count = sum(1 for m in marks if str(m) in MARKED_SET)
                    top_count = sum(1 for x in (a, b, c) if int(x) in velobi_top4)
                    both_count = sum(
                        1
                        for x, m in zip((a, b, c), marks)
                        if str(m) in MARKED_SET and int(x) in velobi_top4
                    )

                    # 三連複の評価重複は、3車すべてが
                    # 「外部印あり＋順流評価1〜4」を満たす場合だけ採用する。
                    # 例：◎・順流1位 / 〇・順流3位 / △・順流2位 のような形だけ。
                    if both_count < 3:
                        continue

                    sc3 = _myoumi_score_3kei(a, b, c, int(role1), mark_map)

                    # 評価重複は安い本線寄りなので、妙味ptの低さで落とさない。
                    # ソートは「印×評価の重複数」→「印数」→「VeloBi上位数」→「列順」→「妙味pt」。
                    out.append((both_count, marked_count, top_count, i, j, k, float(sc3), a, b, c, marks))

        out.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3], x[4], x[5], -x[6], x[7], x[8], x[9]))
        return [(sc3, a, b, c, marks, marked_count, top_count, both_count) for both_count, marked_count, top_count, _, _, _, sc3, a, b, c, marks in out[:EVAL_OVERLAP_MAX_3KEI]]
    except Exception:
        return []


def _collect_myoumi_pickups(col1_cars, col2_cars, col3_cars, role1, mark_map, rec_order_for_forme=None):
    """
    既存フォメから妙味ピックアップ候補を内部データとして返す。
    戻り値：
      two   = [(score, a, b), ...]          # フォメ列順を保持
      three = [(score, a, b, c), ...]       # フォメ列順を保持
    """
    threshold_2kei = MYOUMI_PASS_THRESHOLD_2KEI
    threshold_3kei = MYOUMI_PASS_THRESHOLD_3KEI
    max_2kei = 3
    max_3kei = 3

    c1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
    c2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
    c3 = [int(x) for x in (col3_cars or []) if str(x).isdigit()]
    r1 = int(role1)

    rec_order = [int(x) for x in (rec_order_for_forme or []) if str(x).isdigit()]
    rec_rank = {int(c): i for i, c in enumerate(rec_order)}
    c1_rank = {int(c): i for i, c in enumerate(c1)}
    c2_rank = {int(c): i for i, c in enumerate(c2)}

    two = []
    for a in c1:
        for b in c2:
            if int(a) == int(b):
                continue
            sc = _myoumi_score_2kei(a, b, r1, mark_map)
            if sc >= threshold_2kei:
                two.append((sc, int(a), int(b)))

    two.sort(key=lambda x: (-x[0], c1_rank.get(x[1], 99), c2_rank.get(x[2], 99), rec_rank.get(x[2], 99)))
    two = two[:max_2kei]

    three = []
    seen_ordered = set()
    for a in c1:
        for b in c2:
            for c in c3:
                if len({int(a), int(b), int(c)}) != 3:
                    continue
                # 三連複なので、同じ3車の並び違いは重複表示しない。
                key = tuple(sorted((int(a), int(b), int(c))))
                if key in seen_ordered:
                    continue
                seen_ordered.add(key)
                sc = _myoumi_score_3kei(a, b, c, r1, mark_map, rec_order)
                if sc >= threshold_3kei:
                    three.append((sc, int(a), int(b), int(c)))

    three.sort(key=lambda x: (-x[0], c1_rank.get(x[1], 99), c2_rank.get(x[2], 99), rec_rank.get(x[3], 99)))
    three = three[:max_3kei]

    return two, three


def _make_myoumi_pickup_block(col1_cars, col2_cars, col3_cars, role1, mark_map, rec_order_for_forme=None):
    """
    既存フォメから、妙味基準を超えた買い目だけを抽出する。
    順位生成・フォメ生成には触らず、表示にだけ使う。

    基準：
    ・実戦買目の通過基準は MYOUMI_PASS_THRESHOLD_* を使う。
    ・2車複は8.5pt以上、三連複は8.0pt以上を標準にする。
    """
    try:
        two, three = _collect_myoumi_pickups(
            col1_cars,
            col2_cars,
            col3_cars,
            role1,
            mark_map,
            rec_order_for_forme,
        )

        lines = [f"【妙味ピックアップ｜2車{MYOUMI_PASS_THRESHOLD_2KEI:.1f}pt以上／三連複{MYOUMI_PASS_THRESHOLD_3KEI:.1f}pt以上】", ""]
        lines.append("2車系：")
        if two:
            for sc, a, b in two:
                lines.append(f"{a}-{b}　{sc:.1f}pt")
        else:
            lines.append("該当なし")

        lines.append("")
        lines.append("3連系：")
        if three:
            for sc, a, b, c in three:
                lines.append(f"{_fmt_triple_display(a, b, c)}　{sc:.1f}pt")
        else:
            lines.append("該当なし")

        return "\n".join(lines)
    except Exception:
        return ""



def _myoumi_zone_label(score: float, threshold: float = 7.5) -> str:
    """
    妙味ポイントの表示区分。
    threshold以上は通過、それ未満は参考扱い。
    """
    try:
        sc = float(score)
    except Exception:
        sc = 0.0
    return "通過" if sc >= float(threshold) else "参考"


def _all_2kei_point_items(col1_cars, col2_cars, role1, mark_map):
    """
    2車系フォメ内の全2車複候補を、妙味pt付きで返す。
    同一2車複が複数方向で出る場合は、ptが高い方向を採用する。
    例：1-2 と 2-1 が両方ある場合、高い方の列順で表示する。
    """
    c1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
    c2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
    best = {}
    order = []

    for i, a in enumerate(c1):
        for j, b in enumerate(c2):
            if int(a) == int(b):
                continue
            key = tuple(sorted((int(a), int(b))))
            sc = _myoumi_score_2kei(int(a), int(b), int(role1), mark_map)
            item = (float(sc), int(a), int(b), i, j)
            if key not in best:
                best[key] = item
                order.append(key)
            else:
                old = best[key]
                # pt優先。同点なら先にフォメで出た方向を残す。
                if (item[0], -item[3], -item[4]) > (old[0], -old[3], -old[4]):
                    best[key] = item

    items = [best[k] for k in order if k in best]
    items.sort(key=lambda x: (-x[0], x[3], x[4], x[1], x[2]))
    return [(sc, a, b) for sc, a, b, _, _ in items]


def _all_3kei_point_items(col1_cars, col2_cars, col3_cars, role1, mark_map, rec_order_for_forme=None):
    """
    三連複フォメ内の全候補を、妙味pt付きで返す。
    三連複は重複なし3車として扱い、初回発生のフォメ列順で表示する。
    """
    c1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
    c2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
    c3 = [int(x) for x in (col3_cars or []) if str(x).isdigit()]
    items = []
    seen = set()

    for i, a in enumerate(c1):
        for j, b in enumerate(c2):
            for k, c in enumerate(c3):
                if len({int(a), int(b), int(c)}) != 3:
                    continue
                key = tuple(sorted((int(a), int(b), int(c))))
                if key in seen:
                    continue
                seen.add(key)
                sc = _myoumi_score_3kei(int(a), int(b), int(c), int(role1), mark_map, rec_order_for_forme)
                items.append((float(sc), int(a), int(b), int(c), i, j, k))

    items.sort(key=lambda x: (-x[0], x[4], x[5], x[6], x[1], x[2], x[3]))
    return [(sc, a, b, c) for sc, a, b, c, _, _, _ in items]


def _make_myoumi_point_block(col1_cars, col2_cars, col3_cars, role1, mark_map, rec_order_for_forme=None):
    """
    検算用の妙味ポイント一覧。
    買い目を直接決める欄ではなく、ヴェロビ的買目の根拠確認用として、
    2車複・三連複のフォメ内候補をすべてpt表示する。
    """
    try:
        threshold_2kei = MYOUMI_PASS_THRESHOLD_2KEI
        threshold_3kei = MYOUMI_PASS_THRESHOLD_3KEI
        two_all = _all_2kei_point_items(col1_cars, col2_cars, role1, mark_map)
        three_all = _all_3kei_point_items(col1_cars, col2_cars, col3_cars, role1, mark_map, rec_order_for_forme)

        lines = [f"【妙味ポイント｜2車{threshold_2kei:.1f}pt以上／三連複{threshold_3kei:.1f}pt以上】", ""]

        lines.append("2車複：")
        if two_all:
            for sc, a, b in two_all:
                lines.append(f"{a}-{b}　{sc:.1f}pt［{_myoumi_zone_label(sc, threshold_2kei)}］")
        else:
            lines.append("該当なし")

        lines.append("")
        lines.append("三連複：")
        if three_all:
            for sc, a, b, c in three_all:
                lines.append(f"{_fmt_triple_display(a, b, c)}　{sc:.1f}pt［{_myoumi_zone_label(sc, threshold_3kei)}］")
        else:
            lines.append("該当なし")

        return "\n".join(lines)
    except Exception:
        return ""


def _nishafuku_pairs_from_forme(col1_cars, col2_cars):
    """
    2車系フォメ col1=col2 を2車複の重複なしペアへ変換する。
    例：17=174 → 1-7 / 1-4 / 7-4
    """
    pairs = []
    seen = set()
    c1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
    c2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]

    for a in c1:
        for b in c2:
            if int(a) == int(b):
                continue
            key = tuple(sorted((int(a), int(b))))
            if key in seen:
                continue
            seen.add(key)
            # 表示はフォメに出た順を優先する
            pairs.append((int(a), int(b)))

    return pairs


def _sanpuku_triples_from_forme(col1_cars, col2_cars, col3_cars):
    """
    三連複フォメ col1-col2-col3 を重複なし3車へ変換する。
    表示は最初にフォメで発生した列順を保持する。
    """
    triples = []
    seen = set()
    c1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
    c2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
    c3 = [int(x) for x in (col3_cars or []) if str(x).isdigit()]

    for a in c1:
        for b in c2:
            for c in c3:
                if len({int(a), int(b), int(c)}) != 3:
                    continue
                key = tuple(sorted((int(a), int(b), int(c))))
                if key in seen:
                    continue
                seen.add(key)
                triples.append((int(a), int(b), int(c)))

    return triples


def _fmt_pair(a, b):
    return f"{int(a)}-{int(b)}"


def _fmt_triple(a, b, c):
    return f"{int(a)}-{int(b)}-{int(c)}"


def _triple_display_order(a, b, c):
    """
    三連複は組み合わせ券なので、表示は車番昇順に統一する。
    重要：印・順流順位の注記も、この表示順に合わせて出す。
    これをやらないと、1-7-4 のような列順表示と、
    画面側の 1-4-7 表示を見比べた時に、印と順位がズレて見える。
    """
    try:
        return sorted([int(a), int(b), int(c)])
    except Exception:
        return [int(a), int(b), int(c)]


def _fmt_triple_display(a, b, c):
    x, y, z = _triple_display_order(a, b, c)
    return f"{x}-{y}-{z}"

def _velobi_rank_index(car: int, rec_order_for_forme=None) -> int:
    """順流メイン順のindex。見つからない車は後ろへ回す。"""
    try:
        order = [int(x) for x in (rec_order_for_forme or []) if str(x).isdigit()]
        c = int(car)
        if c in order:
            return order.index(c)
    except Exception:
        pass
    return 999


def _velobi_ordered_cars(cars, rec_order_for_forme=None):
    """評価重複の単系参考用に、車番ではなくVeloBi順流順位で並べる。"""
    try:
        return sorted([int(x) for x in cars], key=lambda x: (_velobi_rank_index(x, rec_order_for_forme), int(x)))
    except Exception:
        return [int(x) for x in cars]


def _fmt_nitan_reference(a, b, rec_order_for_forme=None):
    """2車複の評価重複を、VeloBi順の2車単参考表記に変換する。"""
    x, y = _velobi_ordered_cars([a, b], rec_order_for_forme)
    return f"{x}→{y}"


def _fmt_santan_reference(a, b, c, rec_order_for_forme=None):
    """三連複の評価重複を、VeloBi順の3連単参考表記に変換する。"""
    x, y, z = _velobi_ordered_cars([a, b, c], rec_order_for_forme)
    return f"{x}→{y}→{z}"


def _pair_overlap_note_ordered(a, b, mark_map, rec_order_for_forme=None, top_n: int = 4):
    """2車単参考表記の順番に合わせて注記を並べる。"""
    cars = _velobi_ordered_cars([a, b], rec_order_for_forme)
    return "/".join([
        _overlap_note_for_car(x, mark_map, rec_order_for_forme, top_n=top_n)
        for x in cars
    ])


def _triple_overlap_note_ordered(a, b, c, mark_map, rec_order_for_forme=None, top_n: int = 4):
    """3連単参考表記の順番に合わせて注記を並べる。"""
    cars = _velobi_ordered_cars([a, b, c], rec_order_for_forme)
    return "/".join([
        _overlap_note_for_car(x, mark_map, rec_order_for_forme, top_n=top_n)
        for x in cars
    ])


def _triple_overlap_note(a, b, c, mark_map, rec_order_for_forme=None, top_n: int = 4):
    """三連複の表示順と注記順を必ず一致させる。"""
    cars = _triple_display_order(a, b, c)
    return "/".join([
        _overlap_note_for_car(x, mark_map, rec_order_for_forme, top_n=top_n)
        for x in cars
    ])



def _fmt_cars_compact_for_forme(cars):
    """車番リストをフォメ用の連結文字へ。例：[2,5] -> 25"""
    try:
        xs = []
        for x in cars or []:
            xi = int(x)
            if xi not in xs:
                xs.append(xi)
        return "".join(str(x) for x in xs) if xs else "—"
    except Exception:
        return "—"


# ==============================
# 三展開合成フォメ 圧縮設定
# ==============================
ATTACK_FORME_MAX_TICKETS = 3   # 最終購入点数。v76は三連単1-23-24の3点。
ATTACK_FORME_MAX_SECONDS = 2   # 2列目最大。A-BC-CD型のBC。
ATTACK_FORME_MAX_THIRDS  = 2   # 3列目最大。v76は A-BC-BD 型。
MATERIAL_FORME_MAX_THIRDS = 2   # 素材三連複フォメの3列目最大。超過分は4列目へ分離。


def _expand_santan_forme(A, seconds, thirds):
    """
    A-BC-CD 型を三連単の実買い目へ展開する。
    同一車番重複は自然除外する。
    例：5-43-36 -> 5→4→3 / 5→4→6 / 5→3→6
    """
    out = []
    try:
        A = int(A)
        for s in seconds or []:
            for t in thirds or []:
                s = int(s)
                t = int(t)
                if len({A, s, t}) != 3:
                    continue
                out.append(f"{A}→{s}→{t}")
    except Exception:
        pass
    return out


def _compress_attack_forme(A, seconds, thirds, rec_order_for_forme=None, max_tickets=None):
    """
    最終購入用の三展開合成フォメへ圧縮する。

    目的：
      ・素材フォメをそのまま11点などに広げない。
      ・最終購入は原則3点。
      ・A-BC-CD 型を優先する。
      ・重複除外による自然な3点化を活かす。
    """
    try:
        max_tickets = int(max_tickets or ATTACK_FORME_MAX_TICKETS)
        A = int(A)

        sec = []
        for x in seconds or []:
            xi = int(x)
            if xi != A and xi not in sec:
                sec.append(xi)

        th = []
        for x in thirds or []:
            xi = int(x)
            if xi != A and xi not in th:
                th.append(xi)

        sec = sorted(sec, key=lambda z: (_velobi_rank_index(z, rec_order_for_forme), z))
        th = sorted(th, key=lambda z: (_velobi_rank_index(z, rec_order_for_forme), z))

        sec = sec[:ATTACK_FORME_MAX_SECONDS]
        th = th[:ATTACK_FORME_MAX_THIRDS]

        while len(_expand_santan_forme(A, sec, th)) > max_tickets:
            # 3列目を削る方が、2列目の攻め筋を残しやすい。
            if len(th) > 1:
                th.pop()
            elif len(sec) > 1:
                sec.pop()
            else:
                break

        expanded = _expand_santan_forme(A, sec, th)
        if not expanded:
            return None

        return {
            "forme": f"{A}-{_fmt_cars_compact_for_forme(sec)}-{_fmt_cars_compact_for_forme(th)}",
            "expanded": expanded,
            "seconds": sec,
            "thirds": th,
        }
    except Exception:
        return None


def _line_members_for_car_from_members(line_members_all, car):
    """ライン配列 [[1,2],[3,4]] から指定車のラインを返す。"""
    try:
        c = int(car)
        for mem in line_members_all or []:
            xs = [int(x) for x in mem if str(x).isdigit()]
            if c in xs:
                return xs
    except Exception:
        pass
    return []




def _calc_santen_score_order(style_seq_map=None, active_cars=None, ko_order_for_tie=None, ko_score_map=None):
    """
    順流・渦・逆流の3展開から三展スコア順位を作る。
    点数は各展開の順位を上位ほど加点（7車なら1位=7点）。

    v75:
      三展スコアにKO使用スコアの実数を加算する。
      例：三展19点 + KO2.029 = 21.029
      これにより三展同点を自然に割り、KO上位を軽く反映する。
    """
    try:
        smap = style_seq_map or globals().get("STYLE_SEQ_MAP", {}) or {}
        seqs = []
        for k in ("順流", "渦", "逆流"):
            xs = [int(x) for x in (smap.get(k, []) or []) if str(x).isdigit()]
            if xs:
                seqs.append(xs)
        if not seqs:
            return [], {}, {}

        cars = []
        if active_cars:
            cars = [int(x) for x in active_cars if str(x).isdigit()]
        if not cars:
            for xs in seqs:
                for x in xs:
                    if int(x) not in cars:
                        cars.append(int(x))

        n = max([len(xs) for xs in seqs] + [len(cars), 1])
        santen_score = {int(c): 0.0 for c in cars}
        for xs in seqs:
            for i, c in enumerate(xs):
                c = int(c)
                santen_score.setdefault(c, 0.0)
                santen_score[c] += float(n - i)

        kmap = ko_score_map or globals().get("KO_SCORE_MAP_FOR_SANTEN", {}) or {}
        ko_score = {}
        for c in cars:
            try:
                ko_score[int(c)] = float(kmap.get(int(c), 0.0))
            except Exception:
                ko_score[int(c)] = 0.0

        # --------------------------------------------------
        # v85: 会場判定・最終H補正倍率・必要オッズ倍率をすべて
        #      「三展+KO順位生成」へ反映する。
        #
        # 思想：
        #   ・会場判定        = その会場で素直決着をどれだけ疑うか
        #   ・最終H補正倍率  = H1番手減点/H2番手加点をどれだけ強く見るか
        #   ・必要オッズ倍率 = 安い順当筋を嫌い、補正後順位をどれだけ重視するか
        #
        # これらを order_pressure にまとめ、
        #   三展順位 × (1-order_pressure) + KO/H補正順位 × order_pressure
        # で、最終のA/B/Cを作る。
        # --------------------------------------------------
        try:
            min_odds_mult = float(globals().get("venue_min_odds_mult", 1.0) or 1.0)
        except Exception:
            min_odds_mult = 1.0

        try:
            venue_profile_for_order = str(globals().get("venue_profile", "unknown") or "unknown")
        except Exception:
            venue_profile_for_order = "unknown"

        try:
            home_flow_mult_for_order = float(globals().get("venue_home_flow_mult", 1.0) or 1.0)
        except Exception:
            home_flow_mult_for_order = 1.0

        profile_pressure_map = {
            "strong_good": 0.00,
            "swing_return": 0.18,
            "normal": 0.10,
            "normal_watch": 0.32,
            "cheap_hit": 0.28,
            "bad": 0.50,
            "low_hit_risk": 0.60,
            "very_bad": 0.78,
            "unknown": 0.10,
        }

        odds_pressure = clamp((min_odds_mult - 1.00) / 0.40, 0.0, 1.0)
        profile_pressure = float(profile_pressure_map.get(venue_profile_for_order, 0.10))
        home_pressure = clamp((home_flow_mult_for_order - 1.00) / 0.80, 0.0, 1.0)

        # 必要オッズを主、会場判定を準主、H倍率を補助にする。
        # low_hit_risk / H1.39 / 必要1.30 なら、おおむね0.66前後になる。
        odds_blend = clamp(
            0.50 * float(odds_pressure)
            + 0.35 * float(profile_pressure)
            + 0.15 * float(home_pressure),
            0.0,
            1.0,
        )

        santen_order = sorted(
            [int(c) for c in cars],
            key=lambda c: (-float(santen_score.get(int(c), 0.0)), int(c))
        )
        santen_rank_score = {int(c): float(len(cars) - i) for i, c in enumerate(santen_order)}

        ko = [int(x) for x in (ko_order_for_tie or globals().get("KO_SCORE_ORDER_FOR_TIE", []) or []) if str(x).isdigit()]
        if not ko:
            ko = sorted(
                [int(c) for c in cars],
                key=lambda c: (-float(ko_score.get(int(c), 0.0)), int(c))
            )
        ko_rank = {int(c): i for i, c in enumerate(ko)}
        ko_rank_score = {int(c): float(len(cars) - ko_rank.get(int(c), len(cars))) for c in cars}

        total_score = {}
        for c in cars:
            c = int(c)
            # 同点の微差だけ raw KO を足す。順位の主役は「三展順位×KO/H順位のブレンド」。
            total_score[c] = (
                (1.0 - odds_blend) * float(santen_rank_score.get(c, 0.0))
                + odds_blend * float(ko_rank_score.get(c, 0.0))
                + 0.001 * float(ko_score.get(c, 0.0))
            )

        order = sorted(
            total_score.keys(),
            key=lambda c: (-total_score.get(int(c), 0.0), ko_rank.get(int(c), 999), int(c))
        )
        detail = {
            "santen": santen_score,
            "ko": ko_score,
            "total": total_score,
            "santen_rank_score": santen_rank_score,
            "ko_rank_score": ko_rank_score,
            "venue_min_odds_mult": min_odds_mult,
            "venue_profile": venue_profile_for_order,
            "venue_home_flow_mult": home_flow_mult_for_order,
            "odds_pressure": odds_pressure,
            "profile_pressure": profile_pressure,
            "home_pressure": home_pressure,
            "odds_blend": odds_blend,
        }
        return order, total_score, detail
    except Exception:
        return [], {}, {}



def _make_recommended_flow_34_12_block():
    """
    v89:
    旧「三展開合成フォメ」の場所だけを置き換える表示ブロック。
    他の出力（推奨戦法、メイン着順予想、ヴェロビ的買目、妙味通過、評価重複、期待値推奨）は消さない。

    推奨流れ seq を
      1位=A, 2位=B, 3位=C, 4位=D
    として、34-12 の2車複だけを出す。
      C=A, C=B, D=A, D=B
    つまり 1=2 と 3=4 は買わない。
    """
    try:
        seq = globals().get("RECOMMENDED_STYLE_SEQ", []) or []
        style = str(globals().get("RECOMMENDED_STYLE", "推奨流れ") or "推奨流れ")

        if not seq:
            style_map = globals().get("STYLE_SEQ_MAP", {}) or {}
            seq = style_map.get(style, []) or []

        xs = []
        seen = set()
        for x in seq:
            if str(x).isdigit():
                c = int(x)
                if c not in seen:
                    seen.add(c)
                    xs.append(c)

        if len(xs) < 4:
            return ""

        A, B, C, D = xs[0], xs[1], xs[2], xs[3]
        raw_pairs = [(C, A), (C, B), (D, A), (D, B)]

        pairs = []
        keys = set()
        for a, b in raw_pairs:
            if int(a) == int(b):
                continue
            key = tuple(sorted((int(a), int(b))))
            if key in keys:
                continue
            keys.add(key)
            pairs.append((int(a), int(b)))

        if not pairs:
            return ""

        lines = []
        lines.append("【推奨流れ 34-12 2車複フォメ】")
        lines.append("")
        # noteコピー整理で下部重複ブロックと誤判定されないよう、
        # ここでは plain な「推奨戦法：」は使わない。
        lines.append(f"対象戦法：{style}")
        lines.append("推奨流れ：" + " → ".join(str(int(x)) for x in xs))
        lines.append("")
        for a, b in pairs:
            lines.append(f"2車複｜{a}={b}")
        return "\n".join(lines)
    except Exception as e:
        return f"【推奨流れ 34-12 2車複フォメ】生成不可（{e}）"


def _make_recommended_flow_12_all_trio_switch_block():
    """
    画面表示用の短縮ブロック。
    実効オッズ入力や12-34側の確率計算は使わない。
    1-2ワイド確率から出した推奨下限と、短い条件・買い目だけを出す。
    """
    try:
        seq = globals().get("RECOMMENDED_STYLE_SEQ", []) or []
        style = str(globals().get("RECOMMENDED_STYLE", "推奨流れ") or "推奨流れ")

        if not seq:
            style_map = globals().get("STYLE_SEQ_MAP", {}) or {}
            seq = style_map.get(style, []) or []

        xs = []
        seen = set()
        for x in seq:
            if str(x).isdigit():
                c = int(x)
                if c not in seen:
                    seen.add(c)
                    xs.append(c)

        if len(xs) < 3:
            return ""

        A, B = int(xs[0]), int(xs[1])
        rest = [int(x) for x in xs[2:] if int(x) not in (A, B)]
        if not rest:
            return ""

        stats = globals().get("FLOW_SWITCH_STATS", None) or _get_flow_switch_stats_from_state()
        trio = stats.get("trio12_all", {}) or {}
        pairs = _make_flow_switch_pairs(xs)
        rest_text = "".join(str(int(x)) for x in rest)

        lines = []
        lines.append("【ヴェロビ三連複推奨】")
        lines.append("")
        total_count = stats.get("total_count")
        wide_hits = stats.get("wide12_hits")
        # レース本文は短くする。集計説明は出さない。
        lines.append("条件：")
        lines.append(_flow12_market_nifuku_condition_lines(False, stats))
        lines.append("")
        lines.append("三連複：")
        lines.append(f"{A}-{B}-{rest_text}")

        if pairs:
            lines.append("")
            lines.append("【切替候補｜34-12 2車複】")
            lines.append("")
            lines.append("条件：")
            lines.append(_flow12_market_nifuku_condition_lines(True, stats))
            lines.append("")
            lines.append("2車複：")
            for a, b in pairs:
                lines.append(f"{a}={b}")
            lines.append("")
            lines.append("買い基準：")
            lines.append(_flow3412_nifuku_buy_criteria_line(stats))

        return "\n".join(lines)
    except Exception as e:
        return f"【ヴェロビ三連複推奨】生成不可（{e}）"

def _make_santen_score_attack_forme(max_tickets=None):
    """
    三展+KOスコア順位から、最終の三展開合成フォメを作る。

    v81基本形：評価123・安め切りBOX型（5点）
      三連単：A→B→C
      2車単：B→A / C→A
      2車複：A=C / B=C

    役割：
      A→B→C = 本線の一点3連単
      B→A   = 評価2の逆転２車単
      C→A   = 評価3の逆転・回収起爆剤２車単
      A=C   = 評価2飛びの補助２車複
      B=C   = 評価1飛びのズレ補助２車複
    """
    try:
        order, score, detail = _calc_santen_score_order()
        if len(order) < 3:
            return None

        A = int(order[0])
        B = int(order[1])
        C = int(order[2])

        # v82: 1券種1行。
        # 旧表示の「展開：A→B→C / B→A / ...」「抑え2車単」は使わない。
        tickets_lines = [
            f"3連単｜{A}→{B}→{C}　　本線の一点",
            f"2車単｜{B}→{A}　　　評価2の逆転",
            f"2車単｜{C}→{A}　　　評価3の逆転・回収起爆剤",
            f"2車複｜{A}={C}　　　評価2飛びの補助",
            f"2車複｜{B}={C}　　　評価1飛びのズレ補助",
        ]

        lines = []
        lines.append("【三展+KOスコア順位】")
        santen = (detail or {}).get("santen", {})
        ko = (detail or {}).get("ko", {})
        total = (detail or {}).get("total", score)
        odds_blend = float((detail or {}).get("odds_blend", 0.0) or 0.0)
        min_odds_mult = float((detail or {}).get("venue_min_odds_mult", 1.0) or 1.0)
        venue_profile_for_order = str((detail or {}).get("venue_profile", "unknown") or "unknown")
        home_flow_mult_for_order = float((detail or {}).get("venue_home_flow_mult", 1.0) or 1.0)
        odds_pressure = float((detail or {}).get("odds_pressure", 0.0) or 0.0)
        profile_pressure = float((detail or {}).get("profile_pressure", 0.0) or 0.0)
        home_pressure = float((detail or {}).get("home_pressure", 0.0) or 0.0)
        if odds_blend > 0:
            lines.append(
                f"※会場補正を順位へ反映：会場={venue_profile_for_order}／H倍率{home_flow_mult_for_order:.2f}／必要オッズ{min_odds_mult:.2f} "
                f"→ 三展{(1.0-odds_blend)*100:.0f}%＋KO/H{odds_blend*100:.0f}%"
            )
            lines.append(
                f"　内訳：必要{odds_pressure:.2f}・会場{profile_pressure:.2f}・H{home_pressure:.2f}"
            )
        for i, c in enumerate(order, start=1):
            c = int(c)
            lines.append(
                f"{i}位：{c} (合成={total.get(c, 0.0):.3f}｜三展={santen.get(c, 0.0):.1f}+KO={ko.get(c, 0.0):.6f})"
            )

        return {
            "forme": f"{A}-{B}-{C}",
            "expanded": [f"{A}→{B}→{C}", f"{B}→{A}", f"{C}→{A}", f"{A}={C}", f"{B}={C}"],
            "seconds": [B],
            "thirds": [C],
            "nitan_follow": [f"{B}→{A}", f"{C}→{A}"],
            "nitan_forme": f"{B}{C}→{A}",
            "fukusho_pairs": [f"{A}={C}", f"{B}={C}"],
            "santen_order": order,
            "santen_score": score,
            "santen_detail": detail,
            "santen_block": "\n".join(lines),
            "tickets_lines": tickets_lines,
            "tickets_block": "\n".join(tickets_lines),
            "source": "santen_plus_ko_score_yasume_kiri_box_5ten",
        }
    except Exception:
        return None

def _make_pillar_santan_line_forme(overlap_triples, col2_cars, col3_cars, rec_order_for_forme=None, overlap_pairs=None, myoumi_pairs=None):
    """
    評価重複の柱三連単から、ライン補正フォメを作る。

    思想：
      ・評価重複三連単のうち、妙味ptが一番低いものを「一番素直な柱」とみなす。
      ・柱 A→B→C を作る。
      ・残す車番は「柱B・柱C・A同ライン補正」で作る。
      ・ただし、どの列に残すかは基本三連複フォメを優先する。
        - 2列目に存在するものだけを2列目へ残す。
        - 3列目に存在するものだけを3列目へ残す。
      ・つまり A - (残す車番 ∩ 2列目) - (残す車番 ∩ 3列目) の形で出す。

    例1：
      柱 7→5→3、ライン72、2列目251、3列目1346
      → 7-25-3
    例2：
      柱 2→4→1、ライン423、2列目4163、3列目1375
      → 2-413-13
    """
    try:
        c2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
        c3 = [int(x) for x in (col3_cars or []) if str(x).isdigit()]
        # v59: 基本フォメで4列目へ分離した車は、推奨ライン補正フォメの3列目へ戻さない。
        # 例：1-7435-732 / 4列目=56 の場合、ライン補正で 5 をthird_seedに拾っても除外する。
        # v67: 4列目は素材表示専用。
        # 三展開合成フォメへは絶対に副作用を出さない。
        third_exclude = set()
        if not c2 or not c3:
            return None

        # v32：妙味2車複が出た場合は、評価重複より妙味を優先して補正フォメを作る。
        # 妙味2車複が1点だけなら、最低ptの評価重複2車複を合成して3連系の形にする。
        myoumi_pairs = list(myoumi_pairs or [])
        has_myoumi_pillar = bool(myoumi_pairs)

        # 評価重複の「一番数値が低い」ものを柱にする。
        # 三連複評価重複がある場合は A→B→C を柱にする。
        # 無い場合は、評価重複2車複 A→B を柱にし、3列目はライン補正と基本フォメから作る。
        has_triple_pillar = (not has_myoumi_pillar) and bool(overlap_triples)

        # v63: 妙味起点でも、先に A/B/C とライン補正用ヘルパを作る。
        # v62ではこの初期化より前に A や _line_nearest_third_partners を参照し、
        # 推奨フォメブロック自体が消えるケースがあった。
        sc = 0.0
        A = B = C = None

        def _myoumi_init_key(item):
            try:
                scx, ax, bx = float(item[0]), int(item[1]), int(item[2])
                ox = _velobi_ordered_cars([ax, bx], rec_order_for_forme)
                return (-scx, [_velobi_rank_index(z, rec_order_for_forme) for z in ox], ox)
            except Exception:
                return (999.0, [999, 999], [9, 9])

        if has_myoumi_pillar:
            _base = sorted(myoumi_pairs, key=_myoumi_init_key)[0]
            sc, a0, b0 = float(_base[0]), int(_base[1]), int(_base[2])
            A, B = _velobi_ordered_cars([a0, b0], rec_order_for_forme)
            C = None

        note_text_obj = globals().get("note_text", "")
        line_members_all = _parse_line_members_from_note_text(note_text_obj)
        if not line_members_all:
            line_members_all = _line_members_list_from_line_def(globals().get("line_def", {}))

        a_line = _line_members_for_car_from_members(line_members_all, A) if A is not None else []
        a_line_others = [int(x) for x in a_line if A is not None and int(x) != int(A)]

        def _line_nearest_third_partners(car):
            """採用した2着候補の直近の非軸ライン相手だけを返す。"""
            try:
                line = [int(x) for x in _line_members_for_car_from_members(line_members_all, car)]
                ci = int(car)
                if A is None or ci not in line:
                    return []
                idx = line.index(ci)
                for j in range(idx + 1, len(line)):
                    x = int(line[j])
                    if x != int(A) and x != ci:
                        return [x]
                for j in range(idx - 1, -1, -1):
                    x = int(line[j])
                    if x != int(A) and x != ci:
                        return [x]
                return []
            except Exception:
                return []

        if has_myoumi_pillar:
            # v62 固定仕様：
            # ・2列目は最大2車。
            # ・妙味2車複を起点にするが、2車ラインの軸ライン相手だけは強制優先できる。
            #   4車ライン/3車ラインの奥まで「軸ラインだから」で優先しない。
            # ・押し出された妙味相手、評価重複残り、採用2着候補の直近非軸ライン相手だけを3列目へ。
            # ・4列目へ分離した車は推奨3列目へ戻さない。
            keep_set = set()
            second_seed = set()
            third_seed = set()

            def _add_unique(lst, x):
                try:
                    xi = int(x)
                    if xi != int(A) and xi not in lst:
                        lst.append(xi)
                except Exception:
                    pass

            def _myoumi_pair_rank(item):
                try:
                    scx, ax, bx = float(item[0]), int(item[1]), int(item[2])
                    ox = _velobi_ordered_cars([ax, bx], rec_order_for_forme)
                    return (-scx, [_velobi_rank_index(z, rec_order_for_forme) for z in ox], ox)
                except Exception:
                    return (999.0, [999, 999], [9, 9])

            def _overlap_pair_rank(item):
                try:
                    scx, ax, bx = float(item[0]), int(item[1]), int(item[2])
                    ox = _velobi_ordered_cars([ax, bx], rec_order_for_forme)
                    return (scx, [_velobi_rank_index(z, rec_order_for_forme) for z in ox], ox)
                except Exception:
                    return (999.0, [999, 999], [9, 9])

            myoumi_ys = []
            for item in sorted(myoumi_pairs, key=_myoumi_pair_rank):
                try:
                    _sc, _a, _b = float(item[0]), int(item[1]), int(item[2])
                    x, y = _velobi_ordered_cars([_a, _b], rec_order_for_forme)
                    if int(x) == int(A):
                        _add_unique(myoumi_ys, y)
                except Exception:
                    pass

            overlap_ys = []
            if overlap_pairs:
                for item in sorted(overlap_pairs, key=_overlap_pair_rank):
                    try:
                        _sc, _a, _b = float(item[0]), int(item[1]), int(item[2])
                        x, y = _velobi_ordered_cars([_a, _b], rec_order_for_forme)
                        if int(x) == int(A):
                            _add_unique(overlap_ys, y)
                    except Exception:
                        pass

            # 2車ラインの軸相手だけは、妙味よりも「形」を優先して2列目へ入れる候補。
            axis_line_partner = None
            try:
                a_line_xs = [int(x) for x in a_line]
                if len(a_line_xs) == 2:
                    other = [int(x) for x in a_line_xs if int(x) != int(A)]
                    if other and other[0] in c2:
                        axis_line_partner = int(other[0])
            except Exception:
                axis_line_partner = None

            second_list = []
            displaced_for_third = []

            # まず最上位の妙味を1点採用。
            if myoumi_ys:
                _add_unique(second_list, myoumi_ys[0])

            # 2車ラインの軸相手がいれば2列目へ優先採用。
            if axis_line_partner is not None and axis_line_partner not in second_list:
                _add_unique(second_list, axis_line_partner)

            # 残り枠は妙味、なければ評価重複で埋める。
            for y in myoumi_ys[1:]:
                if len(second_list) >= 2:
                    _add_unique(displaced_for_third, y)
                else:
                    _add_unique(second_list, y)
            if len(second_list) < 2:
                for y in overlap_ys:
                    if len(second_list) >= 2:
                        break
                    _add_unique(second_list, y)

            # 2列目の表示順は基本2列目の順に合わせる。
            second_list = [int(x) for x in c2 if int(x) in set(second_list)][:2]

            for y in second_list:
                keep_set.add(int(y))
                second_seed.add(int(y))
                # 採用2着候補自身が基本3列目にもあるなら、入替3着として残す。
                if int(y) in c3 and int(y) not in third_exclude:
                    third_seed.add(int(y))
                # 採用2着候補の直近非軸ライン相手を3列目へ。
                for z in _line_nearest_third_partners(y):
                    zi = int(z)
                    if zi in c3 and zi not in third_exclude:
                        keep_set.add(zi)
                        third_seed.add(zi)

            # 押し出された妙味相手は、基本3列目にいる場合のみ3列目へ。
            for y in displaced_for_third:
                yi = int(y)
                if yi in c3 and yi not in third_exclude:
                    keep_set.add(yi)
                    third_seed.add(yi)

            # 2列目に採用しなかった評価重複相手は、基本3列目にいる場合のみ3列目へ。
            for y in overlap_ys:
                yi = int(y)
                if yi not in second_seed and yi in c3 and yi not in third_exclude:
                    keep_set.add(yi)
                    third_seed.add(yi)

            # 3列目は基本3列目順、最大4車まで。4列目候補は戻さない。
            third_seed = set(int(x) for x in third_seed if int(x) in c3 and int(x) not in third_exclude)
            if len(third_seed) > 4:
                third_seed = set([int(x) for x in c3 if int(x) in third_seed][:4])

        elif has_triple_pillar:
            def _key(item):
                try:
                    sc, a, b, c = float(item[0]), int(item[1]), int(item[2]), int(item[3])
                    ordered = _velobi_ordered_cars([a, b, c], rec_order_for_forme)
                    return (sc, [_velobi_rank_index(x, rec_order_for_forme) for x in ordered], ordered)
                except Exception:
                    return (999.0, [999, 999, 999], [9, 9, 9])

            pillar = sorted(overlap_triples, key=_key)[0]
            sc, a0, b0, c0 = float(pillar[0]), int(pillar[1]), int(pillar[2]), int(pillar[3])
            A, B, C = _velobi_ordered_cars([a0, b0, c0], rec_order_for_forme)

            # Cが3列目にいない形は、既存三連複フォメとつながらないので出さない。
            if int(C) not in c3:
                return None
        else:
            if not overlap_pairs:
                return None

            def _pair_key(item):
                try:
                    sc, a, b = float(item[0]), int(item[1]), int(item[2])
                    ordered = _velobi_ordered_cars([a, b], rec_order_for_forme)
                    return (sc, [_velobi_rank_index(x, rec_order_for_forme) for x in ordered], ordered)
                except Exception:
                    return (999.0, [999, 999], [9, 9])

            pillar = sorted(overlap_pairs, key=_pair_key)[0]
            sc, a0, b0 = float(pillar[0]), int(pillar[1]), int(pillar[2])
            A, B = _velobi_ordered_cars([a0, b0], rec_order_for_forme)
            C = None

        note_text_obj = globals().get("note_text", "")
        line_members_all = _parse_line_members_from_note_text(note_text_obj)
        if not line_members_all:
            line_members_all = _line_members_list_from_line_def(globals().get("line_def", {}))

        a_line = _line_members_for_car_from_members(line_members_all, A)
        a_line_others = [int(x) for x in a_line if int(x) != int(A)]

        def _line_nearest_third_partners(car):
            """
            v62: 採用した2着候補の「直近の非軸ライン相手」だけを3列目へ返す。
            ライン全員は入れない。隣が軸なら、そのさらに隣の非軸車を1車だけ見る。
            例：246で2を2列目 -> 4のみ（6は奥）
                52で5を2列目 -> 2
                146で6を2列目・軸4 -> 1
                416で4を2列目・軸7 -> 1
            """
            try:
                line = [int(x) for x in _line_members_for_car_from_members(line_members_all, car)]
                ci = int(car)
                if ci not in line:
                    return []
                idx = line.index(ci)

                # 後ろ方向を優先。軸は飛ばして、最初の非軸を1車だけ。
                for j in range(idx + 1, len(line)):
                    x = int(line[j])
                    if x != int(A) and x != ci:
                        return [x]

                # 後ろに非軸がなければ前方向。軸は飛ばして、最初の非軸を1車だけ。
                for j in range(idx - 1, -1, -1):
                    x = int(line[j])
                    if x != int(A) and x != ci:
                        return [x]

                return []
            except Exception:
                return []

        # 残す車番：
        # 三連複柱あり：柱B・柱C・A同ライン補正。
        # 三連複柱なし：
        #   ・2列目は「同じ軸Aから出た評価重複2車複の相手」だけを基本に残す。
        #   ・3列目は、その相手のライン後ろ/前、Aライン残りなどを基本フォメの3列目に合わせて残す。
        # これにより、青森4Rのような 4→7 / 4→2 では、
        # 基本フォメ 4-7621-6153 に対して 4-72-15 となる。
        if has_myoumi_pillar:
            keep_set = {int(B)}
            second_seed = set()
            third_seed = set()

            # v40:
            # 妙味2車複が複数出ても、2列目へ全部は置かない。
            # 2列目は妙味ptの高い順から最大2セットまで。
            # 余った妙味相手は、基本フォメ3列目に存在する場合のみ3列目へ回す。
            # 例：静岡4R 妙味 4-6 / 4-5 / 4-1、基本 4-6751-5163
            #   -> 2列目=6,5 ／ 余り1は3列目へ -> 4-65-516
            def _myoumi_pair_rank(item):
                try:
                    scx, ax, bx = float(item[0]), int(item[1]), int(item[2])
                    ox = _velobi_ordered_cars([ax, bx], rec_order_for_forme)
                    # 妙味は高ptを優先し、同点ならVeloBi順。
                    return (-scx, [_velobi_rank_index(z, rec_order_for_forme) for z in ox], ox)
                except Exception:
                    return (999.0, [999, 999], [9, 9])

            myoumi_second_ranked = []
            myoumi_extra_ranked = []
            for item in sorted(myoumi_pairs, key=_myoumi_pair_rank):
                try:
                    _sc, _a, _b = float(item[0]), int(item[1]), int(item[2])
                    x, y = _velobi_ordered_cars([_a, _b], rec_order_for_forme)
                    if int(x) == int(A) and int(y) != int(A):
                        if int(y) not in myoumi_second_ranked and int(y) not in myoumi_extra_ranked:
                            if len(myoumi_second_ranked) < 2:
                                myoumi_second_ranked.append(int(y))
                            else:
                                myoumi_extra_ranked.append(int(y))
                except Exception:
                    pass

            # v61:
            # 妙味2車複が複数ある場合でも、軸ライン相手が基本2列目にいるなら
            # 2列目へ優先採用する。
            # 例：軸7・ライン72・妙味7-4/7-5・基本2列目245なら、
            # 2列目は45ではなく24。押し出された5は3列目へ回し、7-24-51。
            if len(myoumi_second_ranked) >= 2:
                for xi in a_line_others:
                    xi = int(xi)
                    if xi != int(A) and xi in c2 and xi not in myoumi_second_ranked and xi not in third_exclude:
                        displaced = None
                        if len(myoumi_second_ranked) >= 2:
                            displaced = myoumi_second_ranked.pop()
                        myoumi_second_ranked.append(xi)
                        if displaced is not None and displaced not in myoumi_extra_ranked:
                            myoumi_extra_ranked.insert(0, int(displaced))
                        break

            for y in myoumi_second_ranked:
                keep_set.add(int(y))
                second_seed.add(int(y))
                # 選抜した2着候補も、基本3列目にあるなら3着入替を許容。
                if int(y) in c3:
                    third_seed.add(int(y))
                # 2着候補の同ライン残りは3列目候補。
                # v54: ただしライン全員ではなく、直近の相手だけを採用する。
                # 例：246の2を2列目にした場合、4は残すが6は実質4列目扱いで落とす。
                for yi in _line_nearest_third_partners(y):
                    keep_set.add(int(yi))
                    third_seed.add(int(yi))

            # 余った妙味相手は3列目にあれば回す（2列目には増やさない）。
            for y in myoumi_extra_ranked:
                if int(y) in c3:
                    keep_set.add(int(y))
                    third_seed.add(int(y))

            # v61:
            # 妙味複数時の軸ライン相手は、3列目へ無条件追加しない。
            # 基本2列目にいるなら上で2列目へ採用し、いないなら無理に戻さない。

            # 妙味が1点だけの場合は、2列目が薄くなりやすい。
            # v53:
            #   まず「軸Aのライン相手」が基本2列目にいるなら、評価重複より優先して2列目へ足す。
            #   そのうえでまだ2列目が1車だけなら、最低ptの評価重複2車複を合成する。
            #   例：静岡9R 妙味=5-2、軸ライン=51、基本2列目=132
            #      -> 2列目は 1・2 を優先し、5-12-... にする（5-32 にはしない）。
            if len(myoumi_second_ranked) <= 1:
                # 軸ライン相手を2列目へ優先採用（基本2列目にある場合のみ）。
                for xi in a_line_others:
                    xi = int(xi)
                    if xi != int(A) and xi in c2 and xi not in second_seed:
                        keep_set.add(xi)
                        second_seed.add(xi)
                        # 軸ライン相手は2着候補。3着側へは基本フォメにある場合だけ控えめに残す。
                        if xi in c3:
                            third_seed.add(xi)
                        if len(second_seed) >= 2:
                            break

            if len(myoumi_second_ranked) <= 1 and overlap_pairs:
                def _pair_key2(item):
                    try:
                        sc2, a2, b2 = float(item[0]), int(item[1]), int(item[2])
                        ordered2 = _velobi_ordered_cars([a2, b2], rec_order_for_forme)
                        return (sc2, [_velobi_rank_index(x, rec_order_for_forme) for x in ordered2], ordered2)
                    except Exception:
                        return (999.0, [999, 999], [9, 9])

                ranked_overlap_ys = []
                for item in sorted(overlap_pairs, key=_pair_key2):
                    try:
                        _sc, _a, _b = float(item[0]), int(item[1]), int(item[2])
                        x, y = _velobi_ordered_cars([_a, _b], rec_order_for_forme)
                        yi = int(y)
                        if int(x) == int(A) and yi != int(A) and yi not in ranked_overlap_ys:
                            ranked_overlap_ys.append(yi)
                    except Exception:
                        pass

                added_overlap_second = False
                for yi in ranked_overlap_ys:
                    # 軸ライン相手で2列目が2車に達している場合は、評価重複は2列目へ足さず3列目候補へ回す。
                    if yi not in second_seed and not added_overlap_second and len(second_seed) < 2:
                        keep_set.add(yi)
                        second_seed.add(yi)
                        added_overlap_second = True
                        if yi in c3:
                            third_seed.add(yi)
                        # v54: 評価重複を2列目へ足した場合も、同ライン残りは直近相手だけ。
                        for yyi in _line_nearest_third_partners(yi):
                            keep_set.add(int(yyi))
                            third_seed.add(int(yyi))
                    else:
                        # 2列目に足さなかった評価重複相手は、基本3列目にある場合だけ3着へ回す。
                        if yi in c3:
                            keep_set.add(yi)
                            third_seed.add(yi)

            # v52:
            # 妙味2車複が複数ある場合、軸ライン残りを無条件で3列目へ入れると広がりすぎる。
            # 3列目へ回すのは、
            #   1) 採用した2着候補の同ライン残り
            #   2) 余った妙味相手が基本3列目にあるもの
            #   3) 評価重複2車複の相手が基本3列目にあるもの
            # に絞る。
            # 例：静岡8R 妙味=1-7/1-5、評価重複=1→4/1→3、基本=1-4753-73624
            #   -> 2列目=75 ／ 3列目=7・2・3・4（6は入れない）
            if overlap_pairs:
                def _pair_key_myoumi_extra(item):
                    try:
                        scx, ax, bx = float(item[0]), int(item[1]), int(item[2])
                        ox = _velobi_ordered_cars([ax, bx], rec_order_for_forme)
                        return (scx, [_velobi_rank_index(z, rec_order_for_forme) for z in ox], ox)
                    except Exception:
                        return (999.0, [999, 999], [9, 9])
                for item in sorted(overlap_pairs, key=_pair_key_myoumi_extra):
                    try:
                        _sc, _a, _b = float(item[0]), int(item[1]), int(item[2])
                        x, y = _velobi_ordered_cars([_a, _b], rec_order_for_forme)
                        yi = int(y)
                        if int(x) == int(A) and yi != int(A) and yi in c3:
                            keep_set.add(yi)
                            third_seed.add(yi)
                    except Exception:
                        pass

        elif has_triple_pillar:
            keep_set = {int(B), int(C)}
            second_seed = {int(B)}
            third_seed = {int(C)}

            # v35:
            # 妙味が無い評価重複レースでは、2列目に使う2車複重複を
            # 「ptが低い=一番素直な重複」から最大2セットに絞る。
            # それ以外の重複相手は、基本フォメ3列目にあれば3列目へ回す。
            # 例：2→5(2.0), 2→4(3.5), 2→1(4.3) なら
            # 2列目=5,4 ／ 残り1は3列目へ。
            all_pair_ranked = []
            if overlap_pairs:
                def _pair_key3(item):
                    try:
                        sc3, a3, b3 = float(item[0]), int(item[1]), int(item[2])
                        ordered3 = _velobi_ordered_cars([a3, b3], rec_order_for_forme)
                        # 低ptの重複を優先。
                        return (sc3, [_velobi_rank_index(x, rec_order_for_forme) for x in ordered3], ordered3)
                    except Exception:
                        return (999.0, [999, 999], [9, 9])

                for item in sorted(overlap_pairs, key=_pair_key3):
                    try:
                        _sc, _a, _b = float(item[0]), int(item[1]), int(item[2])
                        x, y = _velobi_ordered_cars([_a, _b], rec_order_for_forme)
                        if int(x) == int(A) and int(y) != int(A) and int(y) not in all_pair_ranked:
                            all_pair_ranked.append(int(y))
                    except Exception:
                        pass

            original_pair_count_for_triple = len(all_pair_ranked)

            # 2列目は最大2セットまで。無ければ柱Bを残す。
            pair_second_ranked = all_pair_ranked[:2] if all_pair_ranked else [int(B)]

            # v37:
            # 評価重複2車複が1セットしかない場合は、2列目が薄くなりすぎる。
            # 軸ラインの残りが基本2列目に存在するなら、2列目にも追加する。
            # 例：A=7、評価重複=7→5、軸ライン=72、基本2列目=251
            #   -> 2列目は 5 + 2 = 52
            if len(pair_second_ranked) == 1:
                for x in a_line_others:
                    xi = int(x)
                    if xi != int(A) and xi in c2 and xi not in pair_second_ranked:
                        pair_second_ranked.append(xi)
                        if len(pair_second_ranked) >= 2:
                            break

            for y in pair_second_ranked:
                keep_set.add(int(y))
                second_seed.add(int(y))

            # 余った評価重複相手は3列目にあれば回す。
            for y in all_pair_ranked[2:]:
                if int(y) in c3:
                    keep_set.add(int(y))
                    third_seed.add(int(y))

            # Aラインの残りは3列目候補へ。
            # これにより、軸ラインの後ろ目・三番手目を3着側に残す。
            for x in a_line_others:
                keep_set.add(int(x))
                third_seed.add(int(x))

            # v35では、2着候補Bのライン後ろは自動では足さない。
            # ここを足すと 2-145-136 のように広がりすぎるため、
            # 2-45-146 のように「低pt2セット＋軸ライン残り＋余り重複」へ絞る。

            # あとで2列目表示に使うため、ローカル変数として保持。
            _pair_second_ranked_for_triple = pair_second_ranked
            _original_pair_count_for_triple = original_pair_count_for_triple
        else:
            keep_set = {int(B)}
            second_seed = {int(B)}
            third_seed = set()

            # Aラインの残りは、2着候補ではなく3着候補側へ回す。
            for x in a_line_others:
                keep_set.add(int(x))
                third_seed.add(int(x))

            # 同じ軸Aから出ている評価重複2車複の相手を2列目候補にする。
            # その相手のライン相手は3列目候補にする。
            for item in overlap_pairs or []:
                try:
                    _sc, _a, _b = float(item[0]), int(item[1]), int(item[2])
                    x, y = _velobi_ordered_cars([_a, _b], rec_order_for_forme)
                    if int(x) == int(A) and int(y) != int(A):
                        keep_set.add(int(y))
                        second_seed.add(int(y))
                        y_line = _line_members_for_car_from_members(line_members_all, y)
                        for yy in y_line:
                            if int(yy) != int(A) and int(yy) != int(y):
                                keep_set.add(int(yy))
                                third_seed.add(int(yy))
                except Exception:
                    pass

            # Bの同ライン相手も3列目候補へ。
            b_line = _line_members_for_car_from_members(line_members_all, B)
            for x in b_line:
                if int(x) != int(A) and int(x) != int(B):
                    keep_set.add(int(x))
                    third_seed.add(int(x))

        # 表示順。
        # 三連複柱ありで、同じ軸Aの評価重複2車複が3点以上ある場合は、
        # 2列目をその評価重複2車複の相手順で出す。
        # 例：青森7R 2→1 / 2→4 / 2→5 なら 2-145-...
        # それ以外は従来どおり、基本フォメ2列目の順を守る。
        sec_source = keep_set if has_triple_pillar else second_seed
        pair_ranked = locals().get("_pair_second_ranked_for_triple", []) if has_triple_pillar else []
        if has_triple_pillar and pair_ranked:
            # v36:
            # 三連複柱ありでも、2列目は「低ptの評価重複2車複 最大2セット」に絞る。
            # ただし表示順は基本フォメ2列目の順を守る。
            # 例：基本2列目=4516、低pt2セット=5,4 -> 45
            pair_second_set = {int(x) for x in pair_ranked}
            sec_candidates = [x for x in c2 if int(x) in pair_second_set and int(x) != int(A)]
        else:
            sec_candidates = [x for x in c2 if x in sec_source and x != int(A)]
        if int(B) not in sec_candidates and int(B) != int(A):
            sec_candidates.append(int(B))

        valid_seconds = []
        for s in sec_candidates:
            if int(s) == int(A):
                continue
            if int(s) in c2 or int(s) == int(B):
                valid_seconds.append(int(s))

        valid_seconds = [x for i, x in enumerate(valid_seconds) if x not in valid_seconds[:i]]
        if not valid_seconds:
            return None

        # 3列目：基本三連複フォメ3列目に存在するものだけを残す。
        # 三連複柱ありは keep_set、2車複柱のみは third_seed を中心に使う。
        third_source = keep_set if has_triple_pillar else third_seed
        third_candidates = []
        if C is not None and int(C) != int(A) and int(C) in c3:
            third_candidates.append(int(C))

        if has_triple_pillar:
            # v36:
            # 三連複柱ありは、3列目を「柱C + 軸ライン残り + 余った評価重複相手」に絞る。
            # 基本フォメ3列目だけで絞ると、軸ライン残りが2列目側にしか無いケースで落ちるため、
            # 軸ライン残りは3列目へ明示的に残す。
            # 例：A=2, 軸ライン246, C=1, 低pt2列目=5・4, 余り重複=1 -> 146
            third_pool = []

            # v37:
            # 評価重複2車複が1セットのみで、軸ライン残りを2列目にも足したケースは、
            # 3列目も軸ライン残りを先に見せる。例：7-52-23。
            one_pair_case = (locals().get("_original_pair_count_for_triple", 0) == 1)

            if one_pair_case:
                # 軸ラインの残りを先に3着側へ残す（基本フォメ列に無くても許容）
                for x in a_line_others:
                    if int(x) != int(A):
                        third_pool.append(int(x))
                # 柱C
                if C is not None and int(C) != int(A):
                    third_pool.append(int(C))
            else:
                # 柱C
                if C is not None and int(C) != int(A):
                    third_pool.append(int(C))

                # 軸ラインの残りは3着側へ残す（基本フォメ列に無くても許容）。
                # ただし柱Bそのものは「2着柱」なので、基本3列目に無い限り3列目へ回さない。
                for x in a_line_others:
                    xi = int(x)
                    if xi == int(A):
                        continue
                    if xi == int(B) and xi not in c3:
                        continue
                    third_pool.append(xi)

                # v39:
                # 2列目を「低pt評価重複2車複 最大2セット」に絞った場合、
                # 1セット目（柱B）のライン残りは足しすぎ防止で原則足さない。
                # 2セット目以降の相手については、その同ライン残りが基本3列目にあれば3列目へ残す。
                # 例：静岡1R 4→3 / 4→2、ライン341/25、基本3列目215
                #   2列目=3,2。2セット目の2のライン残り5を3列目へ足し、4-32-215。
                try:
                    for y in list(pair_ranked or [])[1:]:
                        for yi in _line_nearest_third_partners(y):
                            yi = int(yi)
                            if yi in c3:
                                third_pool.append(yi)
                except Exception:
                    pass

            # 余った評価重複相手は、基本フォメ3列目にあれば3着へ回す
            if overlap_pairs:
                try:
                    used_seconds = set(pair_ranked or [])
                    all_pairs_tmp = []
                    def _pair_key4(item):
                        try:
                            sc4, a4, b4 = float(item[0]), int(item[1]), int(item[2])
                            ordered4 = _velobi_ordered_cars([a4, b4], rec_order_for_forme)
                            return (sc4, [_velobi_rank_index(x, rec_order_for_forme) for x in ordered4], ordered4)
                        except Exception:
                            return (999.0, [999, 999], [9, 9])
                    for item in sorted(overlap_pairs, key=_pair_key4):
                        sc4, a4, b4 = float(item[0]), int(item[1]), int(item[2])
                        x4, y4 = _velobi_ordered_cars([a4, b4], rec_order_for_forme)
                        if int(x4) == int(A) and int(y4) != int(A):
                            all_pairs_tmp.append(int(y4))
                    for y in all_pairs_tmp:
                        if int(y) not in used_seconds and int(y) in c3:
                            third_pool.append(int(y))
                except Exception:
                    pass

            # 表示順。
            # v37の1セット補正ケースは、軸ライン残り→柱Cの順を守る。
            # v39: 候補が全て基本3列目に存在する場合は、基本3列目の順を優先する。
            #      例：静岡1R 基本3列目=215 なら 4-32-215。
            #      軸ライン残りなどで基本3列目外の候補が混じる場合はVeloBi順。
            third_pool = [x for i, x in enumerate(third_pool) if x not in third_pool[:i] and int(x) != int(A)]
            if not locals().get("one_pair_case", False):
                if third_pool and all(int(z) in c3 for z in third_pool):
                    c3_order = {int(v): i for i, v in enumerate(c3)}
                    third_pool = sorted(third_pool, key=lambda z: (c3_order.get(int(z), 999), _velobi_rank_index(z, rec_order_for_forme), z))
                else:
                    third_pool = sorted(third_pool, key=lambda z: (_velobi_rank_index(z, rec_order_for_forme), z))
            for x in third_pool:
                if int(x) not in third_candidates:
                    third_candidates.append(int(x))
        else:
            # v41:
            # 2車複妙味から補正フォメを作る場合、2着候補の同ライン残りは
            # 基本3列目に無くても3列目へ残す。
            # 例：静岡4R 4-5が妙味、5のライン=52、基本=4-6751-5163。
            #   2が基本3列目に無いままだと、5を2着にした時のライン相手が消えるため、
            #   4-65-5162 のようにライン補正として3列目へ追加する。
            for x in c3:
                if int(x) in third_source and int(x) != int(A) and int(x) not in third_candidates:
                    third_candidates.append(int(x))

            if has_myoumi_pillar:
                # 基本3列目外でも、ライン補正で作ったthird_seedは追加する。
                # 表示順はVeloBi順を優先しつつ、既存候補の後ろへ足す。
                extras = []
                for x in third_seed:
                    xi = int(x)
                    if xi == int(A) or xi in third_candidates or xi in third_exclude:
                        continue
                    # 3連単展開側では c2/c3に無い車を弾く処理があるため、
                    # ライン補正で追加した車はここで明示的に候補化しておく。
                    extras.append(xi)
                extras = sorted([x for i, x in enumerate(extras) if x not in extras[:i]], key=lambda z: (_velobi_rank_index(z, rec_order_for_forme), z))
                for x in extras:
                    if int(x) not in third_candidates:
                        third_candidates.append(int(x))

        # v59: 4列目候補は、どの経路で拾われても推奨フォメの3列目から除外する。
        if third_exclude:
            third_candidates = [int(x) for x in third_candidates if int(x) not in third_exclude]

        if not third_candidates and C is not None and int(C) != int(A) and int(C) not in third_exclude:
            third_candidates.append(int(C))

        # 展開表示は同一車重複を除いた実買い目だけ。
        expanded = []
        for s in valid_seconds:
            for t in third_candidates:
                if len({int(A), int(s), int(t)}) != 3:
                    continue
                # 三連複フォメ側に全く出てこない相手は原則出さない。
                # ただしv41では、妙味2車複の2着候補ライン残りとして追加した車は、
                # 基本フォメ外でもライン補正候補として許可する。
                if int(t) not in c2 and int(t) not in c3:
                    if not (has_myoumi_pillar and int(t) in third_seed):
                        continue
                expanded.append(f"{int(A)}→{int(s)}→{int(t)}")

        if not expanded:
            return None

        # v64: 最終購入用に3点へ圧縮。
        # 既存の補正で作った valid_seconds / third_candidates を素材として、
        # A-BC-CD 型の「三展開合成フォメ」に落とす。
        attack = _compress_attack_forme(
            A,
            valid_seconds,
            third_candidates,
            rec_order_for_forme=rec_order_for_forme,
            max_tickets=ATTACK_FORME_MAX_TICKETS,
        )

        # v72:
        # 三展開合成フォメの3列目が2列目のコピーになるケースを補正する。
        # 例：4-71-71 は、2列目=攻め、3列目=受けになっておらず、
        #     的中責任が2列目側に寄りすぎる。
        # 方針：
        #   1) 2列目の後半車を残す（4-71なら1）
        #   2) 軸絡み妙味で2列目に採用されなかった車を足す（例：3）
        #   3) それが無ければ、軸ラインの直近相手を足す（例：6）
        # これにより、妙味型なら 4-71-13、ライン型なら 4-71-16/76 に寄る。
        def _rebuild_attack_thirds_if_copied(_attack):
            try:
                if not _attack:
                    return _attack
                secs0 = [int(x) for x in _attack.get("seconds", [])]
                ths0 = [int(x) for x in _attack.get("thirds", [])]
                if not secs0 or not ths0:
                    return _attack

                # コピー判定：3列目がすべて2列目内なら補正対象。
                if not set(ths0).issubset(set(secs0)):
                    return _attack

                rebuilt = []

                def _add_rebuilt(x):
                    try:
                        xi = int(x)
                        if xi != int(A) and xi not in rebuilt:
                            rebuilt.append(xi)
                    except Exception:
                        pass

                # まず2列目の後半側を3着受けに残す。
                if len(secs0) >= 2:
                    _add_rebuilt(secs0[-1])
                elif secs0:
                    _add_rebuilt(secs0[0])

                # v73:
                # 2列目コピーを直す時、妙味残りを先に入れない。
                # 妙味ptが高くても、三展開で薄い単騎・弱線なら3列目の受けとして危険。
                # まず軸ラインの直近相手を足す。
                # 4617なら軸4の直近相手6。これで 4-71-16 に寄せる。
                if len(rebuilt) < ATTACK_FORME_MAX_THIRDS:
                    try:
                        for y0 in a_line_others:
                            _add_rebuilt(y0)
                            if len(rebuilt) >= ATTACK_FORME_MAX_THIRDS:
                                break
                    except Exception:
                        pass

                # まだ足りない場合だけ、妙味ペアの残りを補助で足す。
                # ただし三展開のVeloBi順でかなり薄い車は採用しない。
                # 7車なら上位5番手以内を目安にする。
                if len(rebuilt) < ATTACK_FORME_MAX_THIRDS:
                    try:
                        myoumi_leftovers = []
                        rank_limit = min(5, len(rec_order_for_forme or []))
                        for item in sorted(myoumi_pairs or [], key=_myoumi_pair_rank):
                            _scx, _ax, _bx = float(item[0]), int(item[1]), int(item[2])
                            x0, y0 = _velobi_ordered_cars([_ax, _bx], rec_order_for_forme)
                            if int(x0) == int(A) and int(y0) not in secs0:
                                if _velobi_rank_index(y0, rec_order_for_forme) < rank_limit:
                                    _add_unique(myoumi_leftovers, y0)
                        for y0 in myoumi_leftovers:
                            _add_rebuilt(y0)
                            if len(rebuilt) >= ATTACK_FORME_MAX_THIRDS:
                                break
                    except Exception:
                        pass

                # それでも足りなければ、元の3列目素材から補う。
                if len(rebuilt) < ATTACK_FORME_MAX_THIRDS:
                    for y0 in third_candidates:
                        _add_rebuilt(y0)
                        if len(rebuilt) >= ATTACK_FORME_MAX_THIRDS:
                            break

                rebuilt = rebuilt[:ATTACK_FORME_MAX_THIRDS]
                if not rebuilt:
                    return _attack

                fixed = _compress_attack_forme(
                    A,
                    secs0,
                    rebuilt,
                    rec_order_for_forme=rec_order_for_forme,
                    max_tickets=ATTACK_FORME_MAX_TICKETS,
                )
                return fixed or _attack
            except Exception:
                return _attack

        attack = _rebuild_attack_thirds_if_copied(attack)

        # v74:
        # 三展開合成フォメは、妙味・ライン補正ではなく三展スコア順位から作る。
        # VeloBi列評価は素材として維持し、ここだけを 1-23-24 型へ差し替える。
        santen_attack = _make_santen_score_attack_forme(max_tickets=ATTACK_FORME_MAX_TICKETS)
        if santen_attack:
            attack = santen_attack

        if attack:
            forme = attack["forme"]
            expanded = attack["expanded"]
            valid_seconds = attack.get("seconds", valid_seconds)
            third_candidates = attack.get("thirds", third_candidates)
        else:
            forme = f"{int(A)}-{_fmt_cars_compact_for_forme(valid_seconds)}-{_fmt_cars_compact_for_forme(third_candidates)}"

        if C is not None:
            pillar_text = f"{int(A)}→{int(B)}→{int(C)}"
        else:
            pillar_text = f"{int(A)}→{int(B)}"
        return {
            "forme": forme,
            "expanded": expanded,
            "pillar": pillar_text,
            "score": sc,
            "source": attack.get("source") if isinstance(attack, dict) and attack.get("source") else ("myoumi" if has_myoumi_pillar else ("triple" if has_triple_pillar else "pair")),
            "santen_block": attack.get("santen_block", "") if isinstance(attack, dict) else "",
            # v79: 三展+KOから作った抑え2車単を表示側へ渡す
            "nitan_forme": attack.get("nitan_forme", "") if isinstance(attack, dict) else "",
            "nitan_follow": attack.get("nitan_follow", []) if isinstance(attack, dict) else [],
        }
    except Exception:
        return None

def _make_rule_buy_block(col1_cars, col2_cars, col3_cars, role1, mark_map, rec_order_for_forme=None):
    """
    現在の実戦ルールに基づく買い目整理。

    方針：
    ・2車複は2層表示にする。
        1) 妙味通過：7.0pt以上。回収率狙い。
        2) 評価重複：外部印とVeloBi列評価が重なる5.0pt以上。的中率補助。
    ・通常三連複は出さない。
      4-1-3 のような上位123評価そのままの買目は、根拠が薄いため廃止。
    ・三連複も2車複と同じ配列で2層表示にする。
        1) 妙味通過：8.0pt以上。回収率狙い。該当なしでも必ず表示。
        2) 評価重複：外部印とVeloBi列評価が重なる的中率補助。
    ・ワイドは現時点では未採用。
    """
    globals()["PILLAR_LINE_FORME_BLOCK"] = ""

    try:
        c1 = [int(x) for x in (col1_cars or []) if str(x).isdigit()]
        c2 = [int(x) for x in (col2_cars or []) if str(x).isdigit()]
        c3 = [int(x) for x in (col3_cars or []) if str(x).isdigit()]

        if not c1 or not c2:
            return ""

        two, three = _collect_myoumi_pickups(
            c1, c2, c3, role1, mark_map, rec_order_for_forme
        )

        # 妙味通過2車複
        pickup_pairs = []
        pickup_pair_keys = set()
        for _, a, b in two:
            key = tuple(sorted((int(a), int(b))))
            if key not in pickup_pair_keys:
                pickup_pair_keys.add(key)
                pickup_pairs.append((int(a), int(b)))

        # 評価重複2車複（妙味通過とは別枠）
        overlap_pairs = _collect_eval_overlap_2kei(
            c1, c2, int(role1), mark_map, exclude_keys=pickup_pair_keys, rec_order_for_forme=rec_order_for_forme
        )

        # 通常三連複は廃止。
        # 代わりに、1列目-2列目-3列目の中で「評価がかぶる三連複」だけを別枠で出す。
        center_triples = []
        center_keys = set()
        overlap_triples = _collect_eval_overlap_3kei(c1, c2, c3, int(role1), mark_map, rec_order_for_forme)

        # v92: 旧「三展開合成フォメ」の差し替えブロックを、
        #      基本推奨1-2-全三連複＋条件付き34-12切替として出す。
        flow_34_12_block = _make_recommended_flow_12_all_trio_switch_block()
        globals()["PILLAR_LINE_FORME_BLOCK"] = flow_34_12_block

        lines = []
        if flow_34_12_block:
            lines.append(flow_34_12_block)
            lines.append("")
            lines.append("＊＊＊＊")
            lines.append("")

        lines.append("【ヴェロビ的買目】")

        lines.append("")
        lines.append(f"2車複｜妙味通過（{MYOUMI_PASS_THRESHOLD_2KEI:.1f}pt以上）：")
        if pickup_pairs:
            for a, b in pickup_pairs:
                lines.append(_fmt_pair(a, b))
        else:
            lines.append("該当なし")

        lines.append("")
        lines.append("2車複’｜評価重複（2車単参考・VeloBi順）：")
        if overlap_pairs:
            for sc, a, b, marked_count, top_count in overlap_pairs:
                mark_note = _pair_overlap_note_ordered(a, b, mark_map, rec_order_for_forme, top_n=4)
                lines.append(f"{_fmt_nitan_reference(a, b, rec_order_for_forme)}　{sc:.1f}pt［評価重複｜{mark_note}］")
        else:
            lines.append("該当なし")

        # 2車複と同じ配列で、三連複も「妙味通過」→「評価重複」の順に必ず表示する。
        lines.append("")
        lines.append(f"三連複｜妙味通過（{MYOUMI_PASS_THRESHOLD_3KEI:.1f}pt以上）：")
        if three:
            for sc, a, b, c in three:
                lines.append(f"{_fmt_triple_display(a, b, c)}　{sc:.1f}pt［通過］")
        else:
            lines.append("該当なし")

        lines.append("")
        lines.append("三連複’｜評価重複（3連単参考・VeloBi順）：")
        if overlap_triples:
            for sc, a, b, c, marks, marked_count, top_count, both_count in overlap_triples:
                mark_note = _triple_overlap_note_ordered(a, b, c, mark_map, rec_order_for_forme, top_n=4)
                lines.append(f"{_fmt_santan_reference(a, b, c, rec_order_for_forme)}　{sc:.1f}pt［評価重複｜{mark_note}］")
        else:
            lines.append("該当なし")

        # v92: ここだけ差し替える。
        # 旧「三展開合成フォメ」は出さず、推奨流れから1-2-全三連複を出す。
        # 重要：この下のヴェロビ的買目・妙味通過・評価重複・期待値推奨はそのまま残す。
        globals()["PILLAR_LINE_FORME_BLOCK"] = _make_recommended_flow_12_all_trio_switch_block()

        # --------------------------------------------------
        # 期待値推奨：
        # 2車複通過ペアを含む三連複のうち、
        # 三連複側の妙味ptも通過基準以上のものだけ。
        # 評価重複ペアは「安い本線」なので、期待値三連複の起点にはしない。
        # --------------------------------------------------
        ev_triples = []
        ev_seen = set()

        threshold_3kei = MYOUMI_PASS_THRESHOLD_3KEI
        ev_source_triples = _all_3kei_point_items(c1, c2, c3, role1, mark_map, rec_order_for_forme)

        if pickup_pair_keys and ev_source_triples:
            for sc, a, b, c in ev_source_triples:
                if float(sc) < threshold_3kei:
                    continue

                tri = (int(a), int(b), int(c))
                tset = set(tri)
                tkey = tuple(sorted(tri))

                if tkey in center_keys:
                    continue

                linked = False
                for pkey in pickup_pair_keys:
                    if set(pkey).issubset(tset):
                        linked = True
                        break

                if not linked:
                    continue

                if tkey in ev_seen:
                    continue

                ev_seen.add(tkey)
                ev_triples.append((float(sc), tri))

        if ev_triples:
            lines.append("")
            lines.append(f"【期待値推奨｜的中率低想定｜三連複{MYOUMI_PASS_THRESHOLD_3KEI:.1f}pt以上】")
            lines.append("")
            lines.append("三連複：")
            for sc, (a, b, c) in ev_triples:
                lines.append(f"{_fmt_triple_display(a, b, c)}　{sc:.1f}pt［通過］")

        return "\n".join(lines)

    except Exception:
        return ""



def _display_expect_myoumi_label(label: str) -> str:
    """
    v119: 全体妙味の内部判定は旧ロジックのまま残し、表示だけA/B/Cへ丸める。
    旧「低」→ A
    旧「C」 → A
    旧「B」 → B
    旧「A」 → B
    旧「AA」→ C
    旧「荒」→ C
    """
    s = str(label or "").strip()
    if s in ("低", "C"):
        return "A"
    if s in ("B", "A"):
        return "B"
    if s in ("AA", "荒"):
        return "C"
    if s in ("A", "B", "C"):
        return s
    return "B"


def _display_expect_myoumi_labels_in_text(text: str) -> str:
    """本文内に残る全体妙味表記も、表示だけA/B/Cへ統一する。"""
    def repl(m):
        return "全体妙味：" + _display_expect_myoumi_label(m.group(1))
    return re.sub(r"全体妙味：(AA|A|B|C|荒|低)", repl, str(text))

def _replace_axis_line_to_expect(text: str, label: str) -> str:
    """
    note本文の最初の軸評価行を全体妙味へ置換する。
    ここで表示名を「三連複軸想定着内率」へ変更する。
    ※2車複は複数点表示のため、全体妙味の率は3連複軸の目安として扱う。
    """
    pat = r"軸評価：[A-E](?:☆☆|☆)?［[^］]*］（軸想定2着内率\s*(\d+)%）"

    def repl(m):
        pct = m.group(1) if m and m.lastindex else ""
        rate_txt = f"（三連複軸想定着内率 {pct}%）" if pct else ""
        # ここでは旧判定ラベルのまま差し込む。
        # A/B/Cへの表示変換は _display_expect_myoumi_labels_in_text() で一度だけ行う。
        return f"全体妙味：{str(label or '').strip()}" + rate_txt

    return re.sub(pat, repl, text, count=1)


def _strip_existing_top_summary(text: str) -> str:
    """
    既存の上部サマリーだけ削除する。
    詳細部（デイ/ナイター/ミッドナイト/モーニング 以降）は絶対に残す。
    """
    lines = text.splitlines()
    if not lines:
        return text

    axis_idx = None
    for i, line in enumerate(lines):
        if re.match(r"^(軸評価|期待値軸|全体妙味)：", line):
            axis_idx = i
            break

    if axis_idx is None:
        return text

    # 軸行直後の空行を飛ばす
    s = axis_idx + 1
    while s < len(lines) and lines[s].strip() == "":
        s += 1

    # 既存サマリーがないなら何もしない
    if s >= len(lines) or not lines[s].startswith("✅ 推奨戦法："):
        return text

    # 詳細部の開催区分行までをサマリーとみなして削除
    e = s
    detail_pat = re.compile(r"^(モーニング|デイ|ナイター|ミッドナイト)\s")
    while e < len(lines):
        if detail_pat.match(lines[e]):
            break
        e += 1

    if e >= len(lines):
        # 詳細部が見つからない時は危険なので削らない
        return text

    new_lines = lines[:s] + lines[e:]
    return "\n".join(new_lines)


# -----------------------------------------
# 推奨戦法とメイン着順予想を箱で強調表示
# -----------------------------------------
try:
    _rec_style = globals().get("RECOMMENDED_STYLE", "")
    _rec_seq = globals().get("RECOMMENDED_STYLE_SEQ", [])
    _rec_copy = globals().get("RECOMMENDED_STYLE_COPY", "")

    _rec_seq = [int(x) for x in (_rec_seq or []) if str(x).isdigit()]

    if _rec_style and _rec_seq:
        _rec_display_seq = " → ".join(str(int(x)) for x in _rec_seq)

        # v163: note用コピーエリア上部の青網掛けボックスは表示しない。
        # 推奨戦法・メイン着順・コピー用は note_text 本文側に残す。
        pass

except Exception as _e:
    st.caption(f"推奨戦法表示生成不可：{_e}")


# -----------------------------------------
# 全体妙味＋実車番フォーメーション自動生成
# -----------------------------------------
nishatan_forme_line = ""
sanpuku_forme_line = ""
sanrentan_forme_line = ""
myoumi_pickup_block = ""
column_eval_block = ""
expect_axis_label = "C"
expect_axis_score = None
expect_axis_role_marks = []

try:
    _rec_seq = globals().get("RECOMMENDED_STYLE_SEQ", [])
    _rec_seq = [int(x) for x in (_rec_seq or []) if str(x).isdigit()]
    _line_def = globals().get("line_def", {})

    if len(_rec_seq) >= 3:
        role1 = int(_rec_seq[0])
        role2 = int(_rec_seq[1])
        role3_original = int(_rec_seq[2])

        # =====================================================
        # VeloBi列評価 v8：ライン順位割り振り型
        # 理論の柱：
        #   1列目 = 勝ち負けの軸
        #   2列目 = 2着以内に入る相手
        #   3列目 = 3着内・穴・ライン残り
        #
        # 重要：
        #   ・評価上位123をそのまま2列目に押し込まない
        #   ・各ラインを推奨順で順位付けし、ライン単位で2列目/3列目へ割り振る
        #   ・通常三連複123を組めるよう、順流3番手は3列目にも残す
        #   ・主導ライン3番手は穴が出やすいので、例外的に2列目にも入れてよい
        #   ・妙味ptは列を壊すためではなく、後段の買う/切る検算に使う
        # =====================================================
        rec_order_for_forme = list(_rec_seq)

        # ライン配列は、コピー欄の「ライン ...」を優先して復元する。
        line_members_all = _parse_line_members_from_note_text(note_text)
        if not line_members_all:
            line_members_all = _line_members_list_from_line_def(_line_def)
        ranked_lines = _rank_lines_by_order(line_members_all, rec_order_for_forme)

        # 軸が所属する主導ライン
        eval1_line_members_text = _find_line_members_of_car_from_note_text(note_text, role1)
        eval1_line_members_global = _find_line_members_of_car(_line_def, role1)
        if eval1_line_members_text and int(role1) in [int(x) for x in eval1_line_members_text]:
            eval1_line_members = [int(x) for x in eval1_line_members_text]
        else:
            eval1_line_members = [int(x) for x in (eval1_line_members_global or [])]

        # 保険：ranked_lines側にも軸所属ラインがあればそちらを優先
        for mem in ranked_lines:
            if int(role1) in [int(x) for x in mem]:
                eval1_line_members = [int(x) for x in mem]
                break

        # 軸所属ラインの扱いを修正。
        # 以前は「軸がライン先頭」と決め打ちして mem[1] を番手扱いにしていたため、
        # 別府4Rの 124 のように軸2が番手の場合、スコア2位の1が2列目から落ちていた。
        # 列評価では、軸所属ラインのうち軸以外で順流順位が最も高い車を
        # まず2列目の本線相手として入れる。残りは穴ヒモ/三列目候補に回す。
        eval1_line_others = []
        if eval1_line_members:
            eval1_line_others = [int(x) for x in eval1_line_members if int(x) != int(role1)]

        rec_pos_map = {int(c): i for i, c in enumerate(rec_order_for_forme)}
        eval1_line_others_sorted = sorted(
            eval1_line_others,
            key=lambda x: (rec_pos_map.get(int(x), 999), eval1_line_members.index(int(x)) if int(x) in eval1_line_members else 999, int(x))
        )

        eval1_partner = []
        eval1_thirdplus = []
        if eval1_line_others_sorted:
            eval1_partner = [int(eval1_line_others_sorted[0])]
            eval1_thirdplus = [int(x) for x in eval1_line_others_sorted[1:]]

        col1_cars = _uniq_keep([role1])

        # 2列目：ライン単位の連対候補
        # 1) 軸所属ラインの相方（軸が番手なら先頭、軸が先頭なら番手）
        # 2) 他ラインの代表車（単騎は本人、複数ラインは推奨順最上位）
        # 3) 軸所属ラインの残り後位は穴ヒモとして例外的に追加
        col2_cars = []
        for cand in eval1_partner:
            if cand not in col1_cars and cand not in col2_cars:
                col2_cars.append(cand)

        for mem in ranked_lines:
            mem = [int(x) for x in mem]
            if not mem:
                continue
            if int(role1) in mem:
                continue

            # そのライン内で推奨順が最も高い車を代表にする。
            rep = None
            for cand in rec_order_for_forme:
                cand = int(cand)
                if cand in mem:
                    rep = cand
                    break
            if rep is None:
                rep = int(mem[0])

            if rep not in col1_cars and rep not in col2_cars:
                col2_cars.append(rep)
            if len(col2_cars) >= 3:
                break

        # 軸所属ラインの残りは、穴ヒモ枠として2列目にも残す
        for cand in eval1_thirdplus:
            cand = int(cand)
            if cand not in col1_cars and cand not in col2_cars:
                col2_cars.append(cand)

        # 通常3車＋同ライン穴枠で最大4車まで
        col2_cars = _uniq_keep(col2_cars[:4])

        expect_axis_label, expect_axis_score, expect_axis_role_marks = _calc_expect_axis_score_label(col1_cars, col2_cars, role1, market_mark_map)

        # 3列目：三連複候補 v42
        # 重要修正：2列目に採用した車の同ライン残りは、基本3列目に必ず残す。
        # 例：ライン52で2列目に5を採用したなら、2は基本3列目候補へ入れる。
        # これを落とすと、4-5-2 のようなライン筋が基本フォメから消えてしまう。
        col3_cars = []

        def _add_col3(cand):
            try:
                cand = int(cand)
            except Exception:
                return
            if cand not in col1_cars and cand not in col3_cars:
                col3_cars.append(cand)

        # 1) 順流3番手を最優先で入れる（通常123を組めるようにする）
        _add_col3(role3_original)

        # 2) 軸ラインの残り後位
        for cand in eval1_thirdplus:
            _add_col3(cand)

        # 3) 2列目に採用された車の同ライン残りを必ず3列目へ
        #    ここは基本フォメ側のライン整合性を作るため、通常の4車上限では落とさない。
        for sec in col2_cars:
            sec = int(sec)
            sec_line = None
            for mem in ranked_lines:
                mem_i = [int(x) for x in mem]
                if sec in mem_i:
                    sec_line = mem_i
                    break
            if not sec_line:
                continue
            for mate in sec_line:
                mate = int(mate)
                if mate == int(role1) or mate == sec:
                    continue
                _add_col3(mate)

        # 4) 余白があれば、従来どおり各ラインの残りをライン順位順に補充
        for mem in ranked_lines:
            mem = [int(x) for x in mem]
            if not mem:
                continue
            if int(role1) in mem:
                rest = [int(x) for x in mem[2:]]
            else:
                rep = None
                for cand in rec_order_for_forme:
                    cand = int(cand)
                    if cand in mem:
                        rep = cand
                        break
                if rep is None:
                    rep = int(mem[0])
                if len(mem) == 1:
                    rest = [rep]
                else:
                    rest = [int(x) for x in mem if int(x) != int(rep)]

            for cand in rest:
                _add_col3(cand)
                # 2列目ライン相手を落とさないため、最大5車まで許容
                if len(col3_cars) >= 5:
                    break
            if len(col3_cars) >= 5:
                break

        # なお、2列目との重複は許可する。
        # 例：2着候補でも、三連複の3列目残りにもなり得る。
        col3_cars = _uniq_keep(col3_cars[:5])

        # v69: 4列目分離前の3列目候補を素材表示用に保持する。
        # ここを保持しないと、軸ライン直近相手（例：4617の6）が
        # 先に4列目へ落ちた時点で復帰不能になる。
        col3_cars_before_col4_split = _uniq_keep(col3_cars)

        # v56：4列目を作る。
        # 目的：フォーメーションは「全部を3列目に入れる」ものではない。
        # 4車ライン、または軸ではない3車以上ラインで、ライン内VeloBi評価3番手以降まで
        # 3列目へ一律に残すと、今回のように無駄な買い目が増える。
        # そこで、ライン内の評価順位で3番手以降は4列目へ分離する。
        # 例：1753で軸1なら、ライン内VeloBi順が 1,7,3,5 の場合、3・5は4列目寄り。
        #     426なら、ライン内VeloBi順が 4,2,6 の場合、6は4列目寄り。
        # ただし2車ライン・単騎は対象外。
        col4_cars = []

        def _line_velobi_depth_for_col4(cand):
            try:
                ci = int(cand)
                for mem in ranked_lines:
                    xs = [int(x) for x in mem]
                    if ci not in xs or len(xs) < 3:
                        continue

                    # ライン内のVeloBi評価順。rec_order_for_formeに出る順を優先し、
                    # 同順・欠落時はライン入力順で補正する。
                    line_pos = {int(x): i for i, x in enumerate(xs)}
                    xs_eval = sorted(xs, key=lambda z: (rec_pos_map.get(int(z), 999), line_pos.get(int(z), 999), int(z)))
                    if ci not in xs_eval:
                        continue
                    depth = xs_eval.index(ci) + 1
                    axis_in_line = int(role1) in xs

                    # v58:
                    # 2列目にいるだけでは3列目保護にしない。
                    # 2列目は「連対候補」であって、深いライン位置の3着残りまで
                    # 自動で買う意味ではないため。
                    # ただし、軸との2車複妙味が通過していて、かつ2列目にも採用されている車は
                    # 買い筋として明確なので3列目に残す。
                    in_col2 = ci in [int(x) for x in col2_cars]
                    pair_myoumi_pass = False
                    try:
                        pair_myoumi_pass = (_myoumi_score_2kei(int(role1), ci, int(role1), market_mark_map) >= MYOUMI_PASS_THRESHOLD_2KEI)
                    except Exception:
                        pair_myoumi_pass = False

                    if in_col2 and pair_myoumi_pass:
                        return False

                    # ライン内の評価上位2番手までは3列目に残す。
                    # 3番手以降は、上の明確な妙味保護がなければ4列目へ分離する。
                    if depth <= 2:
                        return False

                    # 4車ラインは軸ラインでも深い。評価3番手以降は原則4列目。
                    if len(xs) >= 4 and depth >= 3:
                        return True

                    # 軸ではない3車以上ラインも、評価3番手以降は原則4列目。
                    if (not axis_in_line) and len(xs) >= 3 and depth >= 3:
                        return True
            except Exception:
                pass
            return False

        col3_main = []
        for cand in col3_cars:
            ci = int(cand)
            if _line_velobi_depth_for_col4(ci):
                if ci not in col4_cars:
                    col4_cars.append(ci)
            else:
                if ci not in col3_main:
                    col3_main.append(ci)

        col3_cars = _uniq_keep(col3_main)
        col4_cars = _uniq_keep(col4_cars)

        # v69：ここから下は「素材表示用」だけを4列化する。
        # 重要：
        #   ・三展開合成フォメ、妙味通過、期待値推奨、妙味ポイントは
        #     従来計算側を使うため、4列目表示の影響を受けない。
        #   ・素材表示の3列目圧縮では、4列目分離前の候補を使う。
        #     これにより、軸ライン直近相手が先に4列目へ落ちても復帰できる。
        #   ・例：ライン4617・軸4なら、6は軸ライン直近相手なので3列目に復帰。
        #     7や3のような末端・弱別線側を4列目へ回しやすくする。
        col3_cars_full_for_calc = _uniq_keep(col3_cars_before_col4_split)
        col4_cars_display = _uniq_keep(col4_cars)

        try:
            _max_third = int(MATERIAL_FORME_MAX_THIRDS)
        except Exception:
            _max_third = 2

        def _axis_nearest_partners_for_material():
            """
            素材表示用の保護候補。
            軸ラインの直近相手だけを3列目優先に残す。
            長いラインの末端を、軸との妙味だけで過保護しないための補正。
            """
            out = []
            try:
                A0 = int(role1)
                for mem in ranked_lines:
                    xs = [int(x) for x in mem]
                    if A0 not in xs or len(xs) < 2:
                        continue
                    idx = xs.index(A0)

                    # 軸の直後を最優先
                    if idx + 1 < len(xs):
                        x = int(xs[idx + 1])
                        if x != A0:
                            out.append(x)

                    # 軸が番手以降にいるケースの直前も一応保護
                    if idx - 1 >= 0:
                        x = int(xs[idx - 1])
                        if x != A0:
                            out.append(x)
                    break
            except Exception:
                pass
            return _uniq_keep(out)

        material_protect = [
            int(x) for x in _axis_nearest_partners_for_material()
            if int(x) in [int(v) for v in col3_cars_full_for_calc]
        ]

        # 表示用3列目は、単純な先頭2車切りではなく、
        # まず軸ライン直近相手を保護し、その後に元のcol3順で補完する。
        col3_cars_display = []
        for x in material_protect:
            if int(x) not in col3_cars_display:
                col3_cars_display.append(int(x))

        # 補完は「2列目と重複しない候補」を優先する。
        # 例：4-7531 の素材3列目なら、1・7よりも2・6を優先しやすくする。
        col2_set_for_material = {int(v) for v in col2_cars}
        fill_pool = [int(x) for x in col3_cars_full_for_calc if int(x) not in col2_set_for_material]
        fill_pool += [int(x) for x in col3_cars_full_for_calc if int(x) in col2_set_for_material]
        fill_pool = _uniq_keep(fill_pool)

        for x in fill_pool:
            if _max_third > 0 and len(col3_cars_display) >= _max_third:
                break
            xi = int(x)
            if xi not in col3_cars_display:
                col3_cars_display.append(xi)

        if _max_third > 0:
            col3_cars_display = col3_cars_display[:_max_third]

        # 表示用3列目に残らなかった候補は4列目へ。
        for x in col3_cars_full_for_calc:
            xi = int(x)
            if xi not in col3_cars_display and xi not in col4_cars_display:
                col4_cars_display.append(xi)

        col3_cars_display = _uniq_keep(col3_cars_display)
        col4_cars_display = _uniq_keep([int(x) for x in col4_cars_display if int(x) not in set(col3_cars_display)])

        # v70：素材表示では「2列目＝信頼」「3列目＝妙味・補完」に分離する。
        # 重要：ここは表示用だけ。三展開合成フォメ・妙味通過・期待値推奨には副作用を出さない。
        try:
            MATERIAL_FORME_MAX_SECONDS = int(globals().get("MATERIAL_FORME_MAX_SECONDS", 2))
        except Exception:
            MATERIAL_FORME_MAX_SECONDS = 2
        try:
            MATERIAL_FORME_MAX_THIRDS_V70 = int(globals().get("MATERIAL_FORME_MAX_THIRDS_V70", 4))
        except Exception:
            MATERIAL_FORME_MAX_THIRDS_V70 = 4

        def _line_members_for_car_material(car):
            try:
                ci = int(car)
                for mem in ranked_lines:
                    xs = [int(x) for x in mem]
                    if ci in xs:
                        return xs
            except Exception:
                pass
            return []

        def _pair_myoumi_score_material(car):
            try:
                return float(_myoumi_score_2kei(int(role1), int(car), int(role1), market_mark_map))
            except Exception:
                return 0.0

        def _add_unique_material(lst, x):
            try:
                xi = int(x)
            except Exception:
                return
            if xi != int(role1) and xi not in lst:
                lst.append(xi)

        axis_line_material = []
        for mem in ranked_lines:
            xs = [int(x) for x in mem]
            if int(role1) in xs:
                axis_line_material = xs
                break

        axis_nearest_material = _axis_nearest_partners_for_material()
        axis_nearest_set_material = {int(x) for x in axis_nearest_material}

        # 2列目候補A：軸ライン内で、直近相手以外の「妙味はあるが末端すぎない」車。
        # 高すぎる妙味を2列目に置きすぎると的中責任が重いので、同条件なら低pt側を信頼寄りにする。
        axis_inner_candidates = []
        if axis_line_material:
            line_pos_material = {int(x): i for i, x in enumerate(axis_line_material)}
            for x in axis_line_material:
                xi = int(x)
                if xi == int(role1) or xi in axis_nearest_set_material:
                    continue
                sc_m = _pair_myoumi_score_material(xi)
                if sc_m >= globals().get("MYOUMI_PASS_THRESHOLD_2KEI", 7.0):
                    axis_inner_candidates.append((sc_m, rec_pos_map.get(xi, 999), line_pos_material.get(xi, 999), xi))
            axis_inner_candidates = sorted(axis_inner_candidates, key=lambda z: (z[0], z[1], z[2], z[3]))

        # 2列目候補B：他ラインの代表。ここは妙味より信頼度を優先し、推奨順の上位を拾う。
        other_line_reps_material = []
        for mem in ranked_lines:
            xs = [int(x) for x in mem]
            if not xs or int(role1) in xs:
                continue
            rep = None
            for cand in rec_order_for_forme:
                ci = int(cand)
                if ci in xs:
                    rep = ci
                    break
            if rep is None:
                rep = int(xs[0])
            # 単騎は2列目信頼枠に置くと妙味寄りになりやすいので、原則後回し。
            line_len = len(xs)
            other_line_reps_material.append((0 if line_len >= 2 else 1, rec_pos_map.get(rep, 999), -line_len, rep))
        other_line_reps_material = sorted(other_line_reps_material, key=lambda z: (z[0], z[1], z[2], z[3]))

        col2_cars_display = []
        # 軸ライン内の妙味信頼候補を1車だけ入れる。例：4617の1。
        if axis_inner_candidates:
            _add_unique_material(col2_cars_display, axis_inner_candidates[0][3])
        # 他ライン代表を1車入れる。例：25の5。
        for _kind, _rp, _llen, rep in other_line_reps_material:
            if len(col2_cars_display) >= MATERIAL_FORME_MAX_SECONDS:
                break
            _add_unique_material(col2_cars_display, rep)

        # 足りない場合だけ、元2列目から信頼寄りを補完。
        if len(col2_cars_display) < MATERIAL_FORME_MAX_SECONDS:
            fill_seconds = sorted(
                [int(x) for x in col2_cars if int(x) != int(role1)],
                key=lambda z: (
                    0 if _pair_myoumi_score_material(z) < globals().get("MYOUMI_PASS_THRESHOLD_2KEI", 7.0) else 1,
                    rec_pos_map.get(int(z), 999),
                    int(z),
                )
            )
            for x in fill_seconds:
                if len(col2_cars_display) >= MATERIAL_FORME_MAX_SECONDS:
                    break
                _add_unique_material(col2_cars_display, x)

        col2_cars_display = _uniq_keep(col2_cars_display[:MATERIAL_FORME_MAX_SECONDS])

        # v71：3列目は「信頼2車＋妙味非単騎＋軸ライン直近相手」を優先する。
        # 目的：妙味が高いだけの弱い単騎が先に入り、軸ライン直近相手を押し出す事故を防ぐ。
        # 例：静岡7R 4617軸4なら、4-15-1576-23 を狙う。
        col3_cars_display_v70 = []
        for x in col2_cars_display:
            _add_unique_material(col3_cars_display_v70, x)

        # 元2列目のうち、妙味通過している車は3列目へ回す。
        # ただし単騎・弱別線は最後に回す。
        myoumi_third_pool_main = []
        myoumi_third_pool_weak = []
        for x in col2_cars:
            xi = int(x)
            if xi == int(role1):
                continue
            sc_m = _pair_myoumi_score_material(xi)
            if sc_m >= globals().get("MYOUMI_PASS_THRESHOLD_2KEI", 7.0):
                line_len = len(_line_members_for_car_material(xi))
                item = (-sc_m, rec_pos_map.get(xi, 999), xi)
                if line_len >= 2:
                    myoumi_third_pool_main.append(item)
                else:
                    myoumi_third_pool_weak.append(item)

        myoumi_third_pool_main = sorted(myoumi_third_pool_main, key=lambda z: (z[0], z[1], z[2]))
        myoumi_third_pool_weak = sorted(myoumi_third_pool_weak, key=lambda z: (z[0], z[1], z[2]))

        # 先に「ラインを持つ妙味」を入れる。例：7。
        for *_rest, x in myoumi_third_pool_main:
            if len(col3_cars_display_v70) >= MATERIAL_FORME_MAX_THIRDS_V70:
                break
            _add_unique_material(col3_cars_display_v70, x)

        # 次に軸ライン直近相手を保護。例：4617の6。
        for x in axis_nearest_material:
            if len(col3_cars_display_v70) >= MATERIAL_FORME_MAX_THIRDS_V70:
                break
            _add_unique_material(col3_cars_display_v70, x)

        # それでも足りなければ、元3列目フル候補から補完。
        for x in col3_cars_full_for_calc:
            if len(col3_cars_display_v70) >= MATERIAL_FORME_MAX_THIRDS_V70:
                break
            _add_unique_material(col3_cars_display_v70, x)

        # 最後に弱い単騎妙味。枠が残った時だけ。
        for *_rest, x in myoumi_third_pool_weak:
            if len(col3_cars_display_v70) >= MATERIAL_FORME_MAX_THIRDS_V70:
                break
            _add_unique_material(col3_cars_display_v70, x)

        col3_cars_display = _uniq_keep(col3_cars_display_v70[:MATERIAL_FORME_MAX_THIRDS_V70])

        # 4列目は、表示素材に出る全候補のうち、2列目・3列目に採用しなかったもの。
        # 弱い別線・末端・妙味だけの車をここへ逃がす。
        all_material_candidates = []
        for seq in (col2_cars, col3_cars_full_for_calc, rec_order_for_forme):
            for x in seq:
                xi = int(x)
                if xi != int(role1):
                    _add_unique_material(all_material_candidates, xi)

        col4_cars_display = []
        used_display = set([int(x) for x in col2_cars_display] + [int(x) for x in col3_cars_display])
        for x in all_material_candidates:
            xi = int(x)
            if xi not in used_display:
                _add_unique_material(col4_cars_display, xi)

        # 4列目はVeloBi順で表示。弱い単騎が先に出て、評価上位薄目が後ろに回る違和感を防ぐ。
        col4_cars_display = sorted(_uniq_keep(col4_cars_display), key=lambda z: (rec_pos_map.get(int(z), 999), int(z)))

        col1_text = _fmt_cars(col1_cars)
        col2_text = _fmt_cars(col2_cars_display)
        col3_text = _fmt_cars(col3_cars_display)
        col4_text = _fmt_cars(col4_cars_display)

        # note上部サマリー用：VeloBi列評価の素材列を保持する。
        globals()["NOTE_COL2_TEXT"] = col2_text
        globals()["NOTE_COL3_TEXT"] = col3_text
        globals()["NOTE_COL4_TEXT"] = col4_text

        column_eval_block = (
            "【VeloBi列評価】\n"
            f"1列目｜軸候補：{col1_text}\n"
            f"2列目｜2車複ヒモ候補：{col2_text}\n"
            f"3列目｜三連複候補：{col3_text}"
        )
        if col4_cars_display:
            column_eval_block += f"\n4列目｜薄目・4着寄り候補：{col4_text}"

        nishatan_points = _count_nishatan(col1_cars, col2_cars_display)
        sanpuku_points = _count_sanpuku(col1_cars, col2_cars_display, col3_cars_display)
        sanrentan_points = _count_sanrentan(col1_cars, col2_cars_display, col3_cars_display)

        nishatan_forme_line = f"2車系フォメ：1列目→2列目 {col1_text}→{col2_text} / {col1_text}={col2_text}（{nishatan_points}点）"
        if col4_cars_display:
            sanpuku_forme_line = f"三連複フォメ：1列目-2列目-3列目-4列目 {col1_text}-{col2_text}-{col3_text}（{sanpuku_points}点）-{col4_text}"
        else:
            sanpuku_forme_line = f"三連複フォメ：1列目-2列目-3列目 {col1_text}-{col2_text}-{col3_text}（{sanpuku_points}点）"
        sanrentan_forme_line = f"3連単フォメ：1列目→2列目→3列目 {col1_text}→{col2_text}→{col3_text}（{sanrentan_points}点）"

        # v59: 上部の推奨ライン補正フォメ生成でも、4列目候補を3列目へ戻さないために共有する。
        # v67: 4列目表示は上部合成フォメへ渡さない。
        globals()["PILLAR_EXCLUDE_THIRD_CARS"] = []

        myoumi_pickup_block = _make_myoumi_pickup_block(
            col1_cars,
            col2_cars,
            col3_cars,
            role1,
            market_mark_map,
            rec_order_for_forme,
        )
        rule_buy_block = _make_rule_buy_block(
            col1_cars,
            col2_cars,
            col3_cars,
            role1,
            market_mark_map,
            rec_order_for_forme,
        )
        myoumi_point_block = _make_myoumi_point_block(
            col1_cars,
            col2_cars,
            col3_cars,
            role1,
            market_mark_map,
            rec_order_for_forme,
        )

        # v163: 全体妙味・旧フォメ・妙味ポイントの青網掛けボックスは表示しない。
        # 必要な情報は note_text / 2車複考察側へ集約する。
        pass
    else:
        nishatan_forme_line = "2車系フォメ：生成不可"
        sanpuku_forme_line = "三連複フォメ：生成不可"
        sanrentan_forme_line = "3連単フォメ：生成不可"

except Exception as _e:
    nishatan_forme_line = f"2車系フォメ：生成不可（{_e}）"
    sanpuku_forme_line = f"三連複フォメ：生成不可（{_e}）"
    sanrentan_forme_line = f"3連単フォメ：生成不可（{_e}）"
    myoumi_pickup_block = ""
    rule_buy_block = ""
    myoumi_point_block = ""
    column_eval_block = ""
    st.caption(nishatan_forme_line)
    st.caption(sanpuku_forme_line)


# -----------------------------------------
# note上部に実戦用サマリーを差し込む
# 詳細部は行単位で保存する
# v94: noteコピペ用は「最終推奨」中心に圧縮する。
#      VeloBi列評価・旧フォメ・妙味ポイント全件・会場H詳細ログはnoteへ出さない。
# -----------------------------------------
def _extract_note_section_lines(block_text: str, header_prefix: str, max_items: int = 3):
    """
    rule_buy_block から指定見出し直下の買い目行だけを抜く。
    note用の補助根拠を短くするための表示専用処理。
    """
    try:
        lines = str(block_text or "").splitlines()
        out = []
        capture = False
        for raw in lines:
            s = raw.strip()
            if not s:
                if capture and out:
                    break
                continue
            if s.startswith(header_prefix):
                capture = True
                continue
            if capture and s.startswith(("2車複｜", "2車複’｜", "三連複｜", "三連複’｜", "【")):
                break
            if capture:
                if s != "該当なし":
                    out.append(s)
                    if len(out) >= int(max_items):
                        break
        return out
    except Exception:
        return []




def _is_car_seri_involved_for_axis(_car):
    try:
        _car = int(_car)
        _seri_comment = globals().get("seri_comment", {}) or {}
        _seri_target = globals().get("seri_target", {}) or {}

        if bool(_seri_comment.get(_car, _seri_comment.get(str(_car), False))):
            return True

        for _src, _dst in (_seri_target or {}).items():
            try:
                if _dst is None or str(_dst).strip() in ("", "None", "—"):
                    continue
                if int(_dst) == _car:
                    return True
            except Exception:
                continue

        return False
    except Exception:
        return False


def _axis_line_follow_summary(_axis):
    try:
        _axis = int(_axis)
        _line_def = globals().get("line_def", {}) or {}
        _trust = globals().get("line_follow_trust", {}) or {}

        _members = []
        for _gid, _mem in (_line_def or {}).items():
            _mm = [int(x) for x in (_mem or []) if str(x).isdigit()]
            if _axis in _mm:
                _members = _mm
                break

        if len(_members) < 3:
            return "ライン後位：3番手以降なし", "normal"

        _thirds = [int(x) for x in _members[2:]]
        _labels = [str(_trust.get(int(x), _trust.get(str(int(x)), "通常")) or "通常") for x in _thirds]

        if any(x in ("流動", "単騎寄り") for x in _labels):
            return f"ライン後位：流動リスクあり（{','.join(str(x) for x in _thirds)}）", "weak"
        if any(x == "地区まとめ" for x in _labels):
            return f"ライン後位：地区まとめで結束弱め（{','.join(str(x) for x in _thirds)}）", "district"
        if any(x == "明確追走" for x in _labels):
            return f"ライン後位：明確追走あり（{','.join(str(x) for x in _thirds)}）", "strong"
        return f"ライン後位：通常追走（{','.join(str(x) for x in _thirds)}）", "normal"

    except Exception:
        return "ライン後位：未判定", "unknown"


def _make_axis_trust_judgement(seq):
    """
    評価1を安心軸にできるかの判定。
    材料：
    ・KO/H補正後の score_map の1位-2位差
    ・評価1/評価2の競り関与
    ・自力/自力自在/自在コメント
    ・3番手以降の追走信頼
    """
    try:
        xs = [int(x) for x in (seq or []) if str(x).isdigit()]
        if len(xs) < 2:
            return {
                "type": "未判定",
                "gap": None,
                "cap": "通常",
                "reasons": ["評価順不足"],
                "line_note": "ライン後位：未判定",
                "line_level": "unknown",
            }

        A, B = int(xs[0]), int(xs[1])
        score_map = globals().get("score_map", {}) or {}
        s1 = float(score_map.get(A, score_map.get(str(A), 0.0)) or 0.0)
        s2 = float(score_map.get(B, score_map.get(str(B), 0.0)) or 0.0)
        gap = s1 - s2

        jiryoku = globals().get("jiryoku_comment", {}) or {}
        jiryoku_jizai = globals().get("jiryoku_jizai_comment", {}) or {}
        jizai = globals().get("jizai_comment", {}) or {}
        single_comment = globals().get("single_comment", {}) or {}
        line_def = globals().get("line_def", {}) or {}

        a_seri = _is_car_seri_involved_for_axis(A)
        b_seri = _is_car_seri_involved_for_axis(B)
        a_jiryoku = bool(jiryoku.get(A, jiryoku.get(str(A), False)))
        a_jiryoku_jizai = bool(jiryoku_jizai.get(A, jiryoku_jizai.get(str(A), False)))
        a_jizai = bool(jizai.get(A, jizai.get(str(A), False)))
        a_single_comment = bool(single_comment.get(A, single_comment.get(str(A), False)))
        if a_jiryoku_jizai or (a_jiryoku and a_jizai):
            a_move_style = "自力自在"
        elif a_jiryoku:
            a_move_style = "自力"
        elif a_jizai:
            a_move_style = "自在"
        else:
            a_move_style = ""
        a_role = role_in_line(A, line_def) if isinstance(line_def, dict) else "single"
        line_note, line_level = _axis_line_follow_summary(A)

        reasons = []
        reasons.append(f"KO差={gap:.3f}")

        if a_seri:
            axis_type = "二強型・見送り寄り"
            cap = "ステップ1まで"
            reasons.append("評価1が競り関与")
        elif gap >= 0.220 and (a_move_style in ("自力", "自力自在", "自在") or a_role == "head"):
            axis_type = "1軸型"
            cap = "ステップ3まで"
            reasons.append("評価1が評価2を明確に上回る")
            if a_move_style:
                reasons.append(f"評価1に{a_move_style}コメント")
            elif a_role == "head":
                reasons.append("評価1がライン先頭")
            if a_single_comment and not a_move_style:
                reasons.append("評価1は単騎コメントのみ")
        elif gap >= 0.160 and b_seri and not a_seri:
            axis_type = "1軸寄り"
            cap = "ステップ2まで"
            reasons.append("評価2が競り関与")
        elif gap <= 0.080:
            axis_type = "混戦寄り"
            cap = "ステップ1まで"
            reasons.append("評価1・2のKO差が小さい")
        else:
            axis_type = "評価1・2二強型"
            cap = "ステップ2まで"
            reasons.append("評価1・2を並列評価")

        if line_level in ("weak", "district") and axis_type in ("1軸型", "1軸寄り"):
            # 評価1本人は軸でも、ライン丸抱えの信頼は落とす
            reasons.append("ライン後位の結束に不安")
            if axis_type == "1軸型":
                cap = "ステップ2まで"

        return {
            "type": axis_type,
            "gap": gap,
            "cap": cap,
            "reasons": reasons,
            "line_note": line_note,
            "line_level": line_level,
        }

    except Exception as e:
        return {
            "type": "未判定",
            "gap": None,
            "cap": "通常",
            "reasons": [f"判定不可:{e}"],
            "line_note": "ライン後位：未判定",
            "line_level": "unknown",
        }

def _make_note_final_summary_block(rec_style, rec_seq, rec_copy, expect_axis_label, rule_buy_block, mark_map=None):
    """
    note貼り付け用の短縮推奨サマリー。

    v117:
    ・三連複は評価1・2を軸に、2列目・3列目とも「全」表示にする。
    ・2列目を評価3までに絞る表示を廃止し、ライン決着も安め上位4点で拾える形にする。
    ・補助2車複は、旧2車複妙味通過＋34-12候補の中から8.5pt以上だけを統合し、説明文なしで短く表示する。
    """
    try:
        xs = []
        seen = set()
        for x in (rec_seq or []):
            if str(x).isdigit():
                c = int(x)
                if c not in seen:
                    seen.add(c)
                    xs.append(c)

        def _pair_display_from_line(s):
            """旧妙味ブロックの行から車番ペアだけを抜く。例：'7-2　9.1pt［通過］' → '7-2'"""
            try:
                m = re.search(r"([1-9])\s*[-=]\s*([1-9])", str(s))
                if not m:
                    return None
                a, b = int(m.group(1)), int(m.group(2))
                if a == b:
                    return None
                return f"{a}-{b}"
            except Exception:
                return None

        def _add_pair(out, pair_keys, a, b):
            try:
                a, b = int(a), int(b)
                if a == b:
                    return
                key = tuple(sorted((a, b)))
                if key in pair_keys:
                    return
                pair_keys.add(key)
                out.append(f"{a}-{b}")
            except Exception:
                pass

        def _safe_col_text(name, fallback):
            try:
                v = str(globals().get(name, "") or "").strip()
                return v if v else fallback
            except Exception:
                return fallback

        def _axis_pair_line_tail_candidates(_a, _b):
            """
            A-Bが同一ラインで並んでいる場合、Bの後ろの3番手以降を
            3着・相手拡張候補として保護する。

            目的：
            ・個人評価が低い3番手でも、A→Bのライン決着では市場上位に残ることがある。
            ・「地区まとめ」は結束弱めだが、ライン残り候補から即消ししない。
            ・「流動」「単騎寄り」は固定ラインとして扱いにくいため保護しない。
            """
            out = []
            try:
                _a, _b = int(_a), int(_b)
                _line_def = globals().get("line_def", {}) or {}
                _trust = globals().get("line_follow_trust", {}) or {}
                _single_comment = globals().get("single_comment", {}) or {}

                for _gid, _mem in (_line_def or {}).items():
                    xs = [int(x) for x in (_mem or []) if str(x).isdigit()]
                    if _a not in xs or _b not in xs:
                        continue
                    ia, ib = xs.index(_a), xs.index(_b)

                    # A→Bの順で隣接しているラインだけを保護対象にする。
                    if ib != ia + 1:
                        continue

                    for x in xs[ib + 1:]:
                        xi = int(x)
                        label = str(_trust.get(xi, _trust.get(str(xi), "通常")) or "通常")
                        single_flag = bool(_single_comment.get(xi, _single_comment.get(str(xi), False)))
                        if label in ("流動", "単騎寄り") or single_flag:
                            continue
                        if xi not in out:
                            out.append(xi)
                    break
            except Exception:
                pass
            return out

        def _merge_car_text(*seqs):
            out = []
            for seq in seqs:
                for x in (seq or []):
                    try:
                        xi = int(x)
                    except Exception:
                        continue
                    if xi not in out:
                        out.append(xi)
            return "".join(str(x) for x in out)

        def _flow_ratio_map_for_trio():
            """
            流れ想定比率。

            v235:
            表示上のライン評価グループ（順流域／渦域／逆流域）で確定した
            3枠のFR比率を最優先で使う。
            これにより、逆流域が空なのに逆流100%などの矛盾を防ぐ。

            フォールバックとして compute_flow_indicators の FR/VTX/U を使う。
            """
            try:
                _zone_ratio = globals().get("FLOW_RATIO_MAP_BY_ZONE", None)
                if isinstance(_zone_ratio, dict):
                    _jr = float(_zone_ratio.get("順流", 0.0) or 0.0)
                    _ur = float(_zone_ratio.get("逆流", 0.0) or 0.0)
                    _vr = float(_zone_ratio.get("渦", 0.0) or 0.0)
                    _zt = _jr + _ur + _vr
                    if _zt > 0:
                        return {"順流": _jr / _zt, "逆流": _ur / _zt, "渦": _vr / _zt}
            except Exception:
                pass

            try:
                _flow = globals().get("_flow", {}) or {}
                _fr = float(_flow.get("FR", 0.0) or 0.0)
                _vtx = float(_flow.get("VTX", 0.0) or 0.0)
                _u = float(_flow.get("U", 0.0) or 0.0)
                _total = _fr + _u + _vtx
                if _total <= 0:
                    return {"順流": 1.0/3.0, "逆流": 1.0/3.0, "渦": 1.0/3.0}
                return {
                    "順流": _fr / _total,
                    "逆流": _u / _total,
                    "渦": _vtx / _total,
                }
            except Exception:
                return {"順流": 1.0/3.0, "逆流": 1.0/3.0, "渦": 1.0/3.0}

        def _fmt_flow_ratio_line(_ratio_map):
            try:
                return (
                    "流れ想定比率】"
                    f"順流{float(_ratio_map.get('順流', 0.0))*100:.0f}%／"
                    f"逆流{float(_ratio_map.get('逆流', 0.0))*100:.0f}%／"
                    f"渦{float(_ratio_map.get('渦', 0.0))*100:.0f}%"
                )
            except Exception:
                return "流れ想定比率】順流—%／逆流—%／渦—%"

        def _fmt_trio_form(_axis, _cols):
            try:
                _axis = int(_axis)
                _cols = [int(x) for x in (_cols or []) if str(x).isdigit() and int(x) != _axis]
                _out = []
                for _x in _cols:
                    if _x not in _out:
                        _out.append(_x)
                return f"{_axis}-{''.join(str(x) for x in _out)}-{''.join(str(x) for x in _out)}"
            except Exception:
                return "該当なし"

        def _trio_form_ticket_count(_cols):
            try:
                _n = len([x for x in (_cols or [])])
                return int(_n * (_n - 1) / 2) if _n >= 2 else 0
            except Exception:
                return 0

        lines = []

        # v193:
        # 買目考察は「推奨戦法1本」ではなく、順流・逆流・渦を全て並列表示する。
        # 会場判定 good/middle/bad による買目切替は廃止し、各流れごとに
        # 「総合評価B以上・総合pt上位2点」の2車複購入候補だけを表示する。
        style_seq_map = globals().get("STYLE_SEQ_MAP", {}) or {}
        if not isinstance(style_seq_map, dict):
            style_seq_map = {}

        flow_items = []
        # v198:
        # 順流・逆流・渦の表示枠は常に残す。
        # 2ライン戦などで独立した逆流シナリオが成立しない場合は、
        # 逆流を削除せず「該当なし」と表示する。
        for _style_name in ["順流", "逆流", "渦"]:
            _seq = style_seq_map.get(_style_name, []) or []
            _flow_xs = []
            _seen_flow = set()
            for _x in (_seq or []):
                if str(_x).isdigit():
                    _c = int(_x)
                    if _c not in _seen_flow:
                        _seen_flow.add(_c)
                        _flow_xs.append(_c)
            flow_items.append((_style_name, _flow_xs))

        # 保険：STYLE_SEQ_MAP が未生成の場合のみ、従来の推奨1本を表示対象にする。
        if not any(_seq for _, _seq in flow_items) and len(xs) >= 3:
            flow_items = [(str(rec_style or "推奨"), list(xs))]

        # v194: 詳細考察の前に、各流れで選ばれた2車複だけを一覧表示するための保持。
        flow_buy_summary = []
        # v201: 採用2点の総合ptも上部サマリーで使うため保持する。
        flow_buy_pt_summary = []

        # v199:
        # 「買目採用」は従来通り各流れの総合B以上・総合pt上位2点。
        # ただし判断材料として、まず各流れの総合B以上候補を全点表示し、
        # その候補同士で複数流れに重複する買目をイチオシとして抽出する。
        flow_b_candidate_summary = []
        # v204: サマリーの本線/抑えは「採用2点」ではなく、総合B以上候補全体から作る。
        # そのため、総合B以上候補のptも保持する。
        flow_b_candidate_pt_summary = []
        # v220: 各流れの「的中順単騎評価」を後で流れ想定比率で加重し、2車複・3連複の共通土台にする。
        flow_hit_avg_summary = []
        # v223: 妙味順単騎評価も同じく流れ比率で加重し、2車複の妙味期待側へ反映する。
        flow_myoumi_avg_summary = []
        # v220: 全21通り2車複評価も流れ別に保持し、加重単騎評価で2車複サマリーを再構成する。
        flow_all_pair_pt_summary = []

        def _append_one_flow_bet_review(_style_name, _seq):
            try:
                _xs = []
                _seen = set()
                for _x in (_seq or []):
                    if str(_x).isdigit():
                        _c = int(_x)
                        if _c not in _seen:
                            _seen.add(_c)
                            _xs.append(_c)

                lines.append(f"【買目考察｜{_style_name}】")
                lines.append("")

                if len(_xs) < 2:
                    flow_buy_summary.append((_style_name, []))
                    flow_buy_pt_summary.append((_style_name, []))
                    flow_b_candidate_summary.append((_style_name, []))
                    flow_b_candidate_pt_summary.append((_style_name, []))
                    flow_hit_avg_summary.append((_style_name, []))
                    flow_myoumi_avg_summary.append((_style_name, []))
                    flow_all_pair_pt_summary.append((_style_name, []))
                    lines.append("該当なし")
                    lines.append("")
                    return

                _A = int(_xs[0])
                _long_span_all_cars = [int(x) for x in _xs if str(x).isdigit()]

                # v196: 流れ別シナリオの主役ラインを、買目評価側にも渡す。
                # ここを使って、順流/渦/逆流ごとの妙味ptに差を付ける。
                try:
                    _scenario_main_line_map = globals().get("STYLE_SCENARIO_MAIN_LINE_MAP", {}) or {}
                    _scenario_main_line = [int(x) for x in (_scenario_main_line_map.get(str(_style_name), []) or []) if str(x).isdigit()]
                except Exception:
                    _scenario_main_line = []
                _scenario_main_set = {int(x) for x in (_scenario_main_line or [])}

                lines.append(f"推奨流れ【{_style_name}】：")
                lines.append(" → ".join(str(int(x)) for x in _xs))
                lines.append("")

                _axis_judge = _make_axis_trust_judgement(_xs)
                _axis_type = str(_axis_judge.get("type", "未判定"))
                _axis_reasons = [str(x) for x in (_axis_judge.get("reasons", []) or [])]
                _axis_line_note = str(_axis_judge.get("line_note", "ライン後位：未判定"))

                lines.append("【軸判定】")
                lines.append(_axis_type)
                if _axis_reasons:
                    lines.append("理由：" + "／".join(_axis_reasons[:4]))
                lines.append(_axis_line_note)
                lines.append("")

                def _longspan_velobi_rank(_car_no):
                    try:
                        return [int(x) for x in _xs].index(int(_car_no)) + 1
                    except Exception:
                        return 99

                def _longspan_velobi_point(_car_no):
                    try:
                        _rank = _longspan_velobi_rank(_car_no)
                        return {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}.get(_rank, 0)
                    except Exception:
                        return 0

                def _longspan_win_point(_car_no):
                    try:
                        _car_no = int(_car_no)
                        _mk = str((mark_map or {}).get(_car_no, (mark_map or {}).get(str(_car_no), "")) or "").strip()
                        _mk = _mk.replace("○", "〇")
                        return {"◎": 4, "〇": 3, "△": 2, "×": 1}.get(_mk, 0)
                    except Exception:
                        return 0

                def _longspan_hit_score_one(_car_no):
                    try:
                        _v = float(_longspan_velobi_point(_car_no))
                        _w = float(_longspan_win_point(_car_no))
                        _bonus = 0.0
                        if _v >= 4 and _w >= 3:
                            _bonus = 1.5
                        elif _v >= 3 and _w >= 2:
                            _bonus = 1.0
                        return 0.6 * _v + 0.4 * _w + _bonus
                    except Exception:
                        return 0.0

                def _longspan_hit_score_pair(_a, _b):
                    try:
                        return float(_longspan_hit_score_one(_a)) + float(_longspan_hit_score_one(_b))
                    except Exception:
                        return 0.0

                def _longspan_hit_rank(_a, _b):
                    try:
                        _s = _longspan_hit_score_pair(_a, _b)
                        if _s >= 10.0:
                            return "A"
                        if _s >= 8.0:
                            return "B"
                        if _s >= 6.0:
                            return "C"
                        return "D"
                    except Exception:
                        return "D"

                def _longspan_myoumi_rank(_score):
                    try:
                        _score = float(_score)
                        if _score >= 10.0:
                            return "A++"
                        if _score >= 9.4:
                            return "A+"
                        if _score >= 8.4:
                            return "A"
                        if _score >= 7.0:
                            return "B"
                        if _score >= 5.5:
                            return "C"
                        return "D"
                    except Exception:
                        return "D"

                def _longspan_myoumi_core_rank(_myoumi_rank):
                    _r = str(_myoumi_rank)
                    if _r in ("A++", "A+", "A"):
                        return "A"
                    if _r in ("B", "C", "D"):
                        return _r
                    return "D"

                def _longspan_total_rank(_hit_rank, _myoumi_rank):
                    _mr = _longspan_myoumi_core_rank(_myoumi_rank)
                    _table = {
                        ("A", "A"): "A", ("A", "B"): "A", ("A", "C"): "B", ("A", "D"): "C",
                        ("B", "A"): "A", ("B", "B"): "B", ("B", "C"): "B", ("B", "D"): "C",
                        ("C", "A"): "B", ("C", "B"): "C", ("C", "C"): "C", ("C", "D"): "D",
                        ("D", "A"): "C", ("D", "B"): "D", ("D", "C"): "D", ("D", "D"): "D",
                    }
                    return _table.get((str(_hit_rank), _mr), "D")

                def _longspan_total_score(_hit_score, _myoumi_score, _hit_rank, _myoumi_rank, _total_rank):
                    """
                    v207:
                    総合ptは、的中点と妙味点のバランスを見るため、加重平均ではなく幾何平均にする。
                    10点満点には丸めず、現状上限（的中12.0・妙味10.8）から最大約11.4点のまま扱う。

                    旧式：0.55 * 的中点 + 0.45 * 妙味点 + ランクボーナス
                    新式：sqrt(的中点 * 妙味点)

                    ※総合ランク表、的中期待ランク、妙味期待ランクは触らない。
                    ※v205の「イチオシ/本線への妙味期待ランク併記」も維持する。
                    """
                    try:
                        _hs = max(0.0, float(_hit_score))
                    except Exception:
                        _hs = 0.0
                    try:
                        _ms = max(0.0, float(_myoumi_score))
                    except Exception:
                        _ms = 0.0
                    try:
                        return round(math.sqrt(_hs * _ms), 1)
                    except Exception:
                        return 0.0

                def _scenario_myoumi_bonus_2kei(_a, _b, _base_score):
                    """
                    v196:
                    市場印だけの妙味ptだと、流れ別シナリオにしても妙味順位がほぼ変わらない。
                    そこで、その流れの主役ラインが絡む2車複へ小幅補正を入れる。

                    ・主役ライン内の2車複：強めに加点
                    ・主役ライン頭候補×高評価別線：中加点
                    ・主役ライン残り×高評価別線：小加点
                    ・主役ライン非関与：微減点

                    これは実オッズではなく、流れ別の仮説妙味を買目表へ反映するための内部pt。
                    """
                    try:
                        a, b = int(_a), int(_b)
                        base = float(_base_score)
                    except Exception:
                        return float(_base_score or 0.0)

                    if not _scenario_main_set:
                        return round(max(0.0, min(10.0, base)), 1)

                    in_a = a in _scenario_main_set
                    in_b = b in _scenario_main_set
                    bonus = 0.0

                    # 主役ライン内決着。2車複では頭裏の順序ブレを吸収できるので最優先で妙味を残す。
                    if in_a and in_b:
                        bonus += 1.25
                    elif in_a or in_b:
                        other = b if in_a else a
                        main_car = a if in_a else b
                        r_other = _longspan_velobi_rank(other)
                        r_main = _longspan_velobi_rank(main_car)

                        # 主役ラインが勝った時に、全体上位が2着へ突っ込む形。
                        if r_other <= 2:
                            bonus += 0.85
                        elif r_other <= 4:
                            bonus += 0.55
                        else:
                            bonus += 0.25

                        # 主役ライン内でも頭候補に近い車を少し優先。
                        if r_main <= 2:
                            bonus += 0.25
                    else:
                        # その流れの主役ラインが絡まない買い目は、比較上少しだけ下げる。
                        bonus -= 0.25

                    return round(max(0.0, min(10.0, base + bonus)), 1)

                def _longspan_pair_sort_key(_row):
                    _rank_order = {"A++": 6, "A+": 5, "A": 4, "B": 3, "C": 2, "D": 1}
                    try:
                        _total_pt = float(_row.get("total_pt", 0.0))
                    except Exception:
                        _total_pt = 0.0
                    try:
                        _hit_score = float(_row.get("hit_score", 0.0))
                    except Exception:
                        _hit_score = 0.0
                    try:
                        _myoumi_score = float(_row.get("myoumi_score", 0.0))
                    except Exception:
                        _myoumi_score = 0.0
                    return (
                        _total_pt,
                        _rank_order.get(str(_row.get("total_rank")), 0),
                        _hit_score,
                        _myoumi_score,
                        _rank_order.get(str(_row.get("hit_rank")), 0),
                        _rank_order.get(str(_row.get("myoumi_rank")), 0),
                    )

                # v203: 会場成績を2車複BOX評価の内部ptへ小幅反映する。
                # 的中率 → 的中期待係数、回収率 → 妙味期待係数。
                # 買目採用ルールは変えず、総合pt計算前の材料だけを補正する。
                try:
                    _venue_hit_coef = float(globals().get("venue_hit_expect_coef", st.session_state.get("venue_hit_expect_coef", 1.00)) or 1.00)
                except Exception:
                    _venue_hit_coef = 1.00
                try:
                    _venue_myoumi_coef = float(globals().get("venue_myoumi_expect_coef", st.session_state.get("venue_myoumi_expect_coef", 1.00)) or 1.00)
                except Exception:
                    _venue_myoumi_coef = 1.00

                _long_span_pairs = []
                _long_span_keys = set()
                for _a, _b in combinations(_long_span_all_cars, 2):
                    try:
                        _a_i, _b_i = int(_a), int(_b)
                        if _a_i == _b_i:
                            continue
                        _key = tuple(sorted((_a_i, _b_i)))
                        if _key in _long_span_keys:
                            continue
                        _long_span_keys.add(_key)

                        _order_pair = sorted([_a_i, _b_i], key=lambda z: _longspan_velobi_rank(z))
                        _score_head, _score_tail = int(_order_pair[0]), int(_order_pair[1])
                        try:
                            _base_sc = float(_myoumi_score_2kei(_score_head, _score_tail, int(_A), mark_map or {}))
                        except Exception:
                            _base_sc = 0.0
                        _sc = _scenario_myoumi_bonus_2kei(_key[0], _key[1], _base_sc)
                        # v203: 回収率が低い開催では妙味期待を少し弱め、回収率が高い開催では少し強める。
                        _sc = round(max(0.0, min(10.8, float(_sc) * float(_venue_myoumi_coef))), 2)

                        _disp = f"{_key[0]}-{_key[1]}"
                        _hit_score = _longspan_hit_score_pair(_key[0], _key[1])
                        # v203: 的中率が低い開催では的中期待を少し弱め、的中率が高い開催では少し強める。
                        _hit_score = round(max(0.0, min(12.0, float(_hit_score) * float(_venue_hit_coef))), 2)
                        if _hit_score >= 10.0:
                            _hit_rank = "A"
                        elif _hit_score >= 8.0:
                            _hit_rank = "B"
                        elif _hit_score >= 6.0:
                            _hit_rank = "C"
                        else:
                            _hit_rank = "D"
                        _myoumi_rank = _longspan_myoumi_rank(_sc)
                        _total_rank = _longspan_total_rank(_hit_rank, _myoumi_rank)
                        _total_pt = _longspan_total_score(_hit_score, _sc, _hit_rank, _myoumi_rank, _total_rank)
                        _long_span_pairs.append({
                            "disp": _disp,
                            "a": _key[0],
                            "b": _key[1],
                            "score_head": _score_head,
                            "score_tail": _score_tail,
                            "hit_rank": _hit_rank,
                            "hit_score": round(float(_hit_score), 2),
                            "myoumi_rank": _myoumi_rank,
                            "myoumi_score": round(float(_sc), 2),
                            "total_rank": _total_rank,
                            "total_pt": _total_pt,
                        })
                    except Exception:
                        pass

                if not _long_span_pairs:
                    flow_buy_summary.append((_style_name, []))
                    flow_buy_pt_summary.append((_style_name, []))
                    flow_b_candidate_summary.append((_style_name, []))
                    flow_b_candidate_pt_summary.append((_style_name, []))
                    flow_hit_avg_summary.append((_style_name, []))
                    flow_myoumi_avg_summary.append((_style_name, []))
                    flow_all_pair_pt_summary.append((_style_name, []))
                    lines.append("該当なし")
                    lines.append("")
                    return

                _sorted_pairs = sorted(_long_span_pairs, key=_longspan_pair_sort_key, reverse=True)
                _nifuku_buy_base = [
                    _row for _row in _sorted_pairs
                    if str(_row.get("total_rank", "")).strip() in ("A", "B")
                ]
                # v210: 流れ内の候補母集団は固定ptで切らず、総合B以上候補を保持する。
                #        冒頭サマリー側で、レース内の上位割合（本線30%・抑え50%）に絞る。
                _nifuku_display_base = list(_nifuku_buy_base or [])
                _b_candidate_disp = [str(_row.get("disp")) for _row in (_nifuku_display_base or []) if _row.get("disp")]
                flow_b_candidate_summary.append((_style_name, list(_b_candidate_disp)))
                flow_b_candidate_pt_summary.append((_style_name, [
                    {
                        "disp": str(_row.get("disp")),
                        "a": int(_row.get("a")),
                        "b": int(_row.get("b")),
                        "hit_score": float(_row.get("hit_score", 0.0) or 0.0),
                        "myoumi_score": float(_row.get("myoumi_score", 0.0) or 0.0),
                        "total_pt": float(_row.get("total_pt", 0.0) or 0.0),
                        "hit_rank": str(_row.get("hit_rank", "") or ""),
                        "myoumi_rank": str(_row.get("myoumi_rank", "") or ""),
                        "total_rank": str(_row.get("total_rank", "") or ""),
                    }
                    for _row in (_nifuku_display_base or []) if _row.get("disp")
                ]))

                flow_all_pair_pt_summary.append((_style_name, [
                    {
                        "disp": str(_row.get("disp")),
                        "a": int(_row.get("a")),
                        "b": int(_row.get("b")),
                        "hit_score": float(_row.get("hit_score", 0.0) or 0.0),
                        "myoumi_score": float(_row.get("myoumi_score", 0.0) or 0.0),
                        "total_pt": float(_row.get("total_pt", 0.0) or 0.0),
                        "hit_rank": str(_row.get("hit_rank", "") or ""),
                        "myoumi_rank": str(_row.get("myoumi_rank", "") or ""),
                        "total_rank": str(_row.get("total_rank", "") or ""),
                    }
                    for _row in (_sorted_pairs or []) if _row.get("disp")
                ]))

                _nifuku_buy = list(_nifuku_display_base or [])[:2]
                _nifuku_buy_disp = [str(_row.get("disp")) for _row in _nifuku_buy if _row.get("disp")]
                flow_buy_summary.append((_style_name, list(_nifuku_buy_disp)))
                flow_buy_pt_summary.append((_style_name, [
                    {"disp": str(_row.get("disp")), "total_pt": float(_row.get("total_pt", 0.0) or 0.0)}
                    for _row in _nifuku_buy if _row.get("disp")
                ]))

                lines.append("【総合評価2車複推奨】")
                lines.append("2車複購入候補（総合B以上・流れ内上位2点）")
                lines.append("　".join(_nifuku_buy_disp) if _nifuku_buy_disp else "該当なし")
                lines.append("")

                def _longspan_trimmed_avg(_vals):
                    try:
                        _vals = sorted([float(v) for v in (_vals or [])])
                        if len(_vals) >= 3:
                            _vals = _vals[1:-1]
                        if not _vals:
                            return 0.0
                        return round(sum(_vals) / len(_vals), 2)
                    except Exception:
                        return 0.0

                def _longspan_car_average_rows(_pairs, _cars):
                    _avg_rows = []
                    try:
                        for _car in [int(x) for x in (_cars or []) if str(x).isdigit()]:
                            _hit_vals = []
                            _myoumi_vals = []
                            _total_vals = []
                            for _row in (_pairs or []):
                                try:
                                    if int(_row.get("a")) == _car or int(_row.get("b")) == _car:
                                        _hit_vals.append(float(_row.get("hit_score", 0.0)))
                                        _myoumi_vals.append(float(_row.get("myoumi_score", 0.0)))
                                        _total_vals.append(float(_row.get("total_pt", 0.0)))
                                except Exception:
                                    pass
                            if _hit_vals and _myoumi_vals and _total_vals:
                                _hit_avg = _longspan_trimmed_avg(_hit_vals)
                                _myoumi_avg = _longspan_trimmed_avg(_myoumi_vals)
                                _avg_rows.append({
                                    "car": _car,
                                    "hit_avg": _hit_avg,
                                    "myoumi_avg": _myoumi_avg,
                                    "total_avg": _longspan_trimmed_avg(_total_vals),
                                })
                    except Exception:
                        _avg_rows = []
                    return _avg_rows

                def _longspan_car_average_line(_avg_rows, _key):
                    try:
                        _rows = sorted(_avg_rows or [], key=lambda r: (float(r.get(_key, 0.0)), -_longspan_velobi_rank(r.get("car"))), reverse=True)
                        return " → ".join(f"{int(r.get('car'))}（{float(r.get(_key, 0.0)):.1f}）" for r in _rows)
                    except Exception:
                        return ""

                _car_avg_rows = _longspan_car_average_rows(_sorted_pairs, _long_span_all_cars)
                flow_hit_avg_summary.append((_style_name, [
                    {"car": int(_r.get("car")), "hit_avg": float(_r.get("hit_avg", 0.0) or 0.0)}
                    for _r in (_car_avg_rows or []) if str(_r.get("car", "")).isdigit()
                ]))
                flow_myoumi_avg_summary.append((_style_name, [
                    {"car": int(_r.get("car")), "myoumi_avg": float(_r.get("myoumi_avg", 0.0) or 0.0)}
                    for _r in (_car_avg_rows or []) if str(_r.get("car", "")).isdigit()
                ]))
                _hit_avg_line = _longspan_car_average_line(_car_avg_rows, "hit_avg")
                _myoumi_avg_line = _longspan_car_average_line(_car_avg_rows, "myoumi_avg")

                lines.append("車番別平均評価（極端値除外）")
                if _hit_avg_line:
                    lines.append(f"的中順単騎評価：{_hit_avg_line}")
                if _myoumi_avg_line:
                    lines.append(f"妙味順単騎評価：{_myoumi_avg_line}")
                lines.append("")

                def _longspan_display_width(_text):
                    import unicodedata
                    _s = str(_text)
                    _w = 0
                    for _ch in _s:
                        _w += 2 if unicodedata.east_asian_width(_ch) in ("F", "W", "A") else 1
                    return _w

                def _longspan_pad_center(_text, _width):
                    _txt = str(_text)
                    _pad = max(0, int(_width) - _longspan_display_width(_txt))
                    _left = _pad // 2
                    _right = _pad - _left
                    return (("　" * (_left // 2)) + (" " * (_left % 2)) +
                            _txt +
                            ("　" * (_right // 2)) + (" " * (_right % 2)))

                _col_w = {
                    "disp": 10,
                    "hit": 10,
                    "myoumi": 10,
                    "total": 10,
                    "pt": 8,
                }
                _sep = ""
                lines.append(_sep.join([
                    _longspan_pad_center("買い目", _col_w["disp"]),
                    _longspan_pad_center("的中期待", _col_w["hit"]),
                    _longspan_pad_center("妙味期待", _col_w["myoumi"]),
                    _longspan_pad_center("総合評価", _col_w["total"]),
                    _longspan_pad_center("総合pt", _col_w["pt"]),
                ]))
                for _row in _sorted_pairs:
                    _disp_cell = _longspan_pad_center(_row.get("disp"), _col_w["disp"])
                    _hit_cell = _longspan_pad_center(_row.get("hit_rank"), _col_w["hit"])
                    _myoumi_cell = _longspan_pad_center(_row.get("myoumi_rank"), _col_w["myoumi"])
                    _total_cell = _longspan_pad_center(_row.get("total_rank"), _col_w["total"])
                    _pt_cell = _longspan_pad_center(f"{float(_row.get('total_pt', 0.0)):.1f}", _col_w["pt"])
                    lines.append(_sep.join([_disp_cell, _hit_cell, _myoumi_cell, _total_cell, _pt_cell]))

                lines.append("")
            except Exception as _e:
                flow_buy_summary.append((_style_name, []))
                flow_buy_pt_summary.append((_style_name, []))
                flow_b_candidate_summary.append((_style_name, []))
                flow_b_candidate_pt_summary.append((_style_name, []))
                flow_hit_avg_summary.append((_style_name, []))
                flow_myoumi_avg_summary.append((_style_name, []))
                flow_all_pair_pt_summary.append((_style_name, []))
                lines.append(f"【買目考察｜{_style_name}】")
                lines.append(f"生成不可（{_e}）")
                lines.append("")

        def _fmt_flow_buy_pairs(_pairs):
            _pairs = [str(x) for x in (_pairs or []) if str(x).strip()]
            return "　".join(_pairs) if _pairs else "該当なし"

        def _flow_summary_label(_style_name):
            # 「渦」は1文字なので、順流/逆流と縦位置が近くなるよう全角空白を足す。
            return "渦　" if str(_style_name) == "渦" else str(_style_name)

        if flow_items:
            # v194: まず詳細を一度組み立て、その過程で flow_buy_summary に各流れの購入候補を保持する。
            _main_lines_ref = lines
            _detail_lines = []
            lines = _detail_lines

            for _i, (_style_name, _seq) in enumerate(flow_items):
                if _i > 0:
                    lines.append("＊＊＊＊")
                    lines.append("")
                _append_one_flow_bet_review(_style_name, _seq)

            lines = _main_lines_ref

            # v210: 冒頭サマリーは固定pt足切りではなく、レース内の順位割合で整理する。
            # 表示順は、
            # 1) 本線：総合B以上候補のうち、総合pt上位30%
            # 2) 抑え：総合B以上候補のうち、総合pt上位50%以内（本線以外）
            # 3) ベスト10内重複：各流れの総合B以上候補・総合pt上位10内で複数流れに重複したもの
            # 4) 流れ別：総合B以上候補のうち、本線/抑えの表示対象に入ったものだけ表示
            # ※各流れ採用2点をサマリーの母集団には使わない。
            _NIFUKU_MAIN_PERCENT = 0.30
            _NIFUKU_DISPLAY_PERCENT = 0.50
            # v222: 流れ加重後は総合B以上が増えやすいので、表示点数に上限を置く。
            # 2車複は購入主役ではなく、3連複の骨格確認用サマリーとして使う。
            _NIFUKU_MAIN_MAX = 3
            _NIFUKU_DISPLAY_MAX = 5
            _summary_map = {}
            _summary_pt_map = {}
            _b_candidate_map = {}
            _overall_pairs = []
            _overall_pair_rows = []
            _overall_seen = set()

            def _pair_key_from_disp(_p):
                try:
                    _m = re.search(r"([1-9])\s*[-=]\s*([1-9])", str(_p))
                    if not _m:
                        return None
                    _a, _b = int(_m.group(1)), int(_m.group(2))
                    return tuple(sorted((_a, _b)))
                except Exception:
                    return None

            def _fmt_flow_buy_pairs(_pairs):
                _pairs = [str(x) for x in (_pairs or []) if str(x).strip()]
                return "　".join(_pairs) if _pairs else "該当なし"

            def _same_pair_list(_a, _b):
                try:
                    _ka = [_pair_key_from_disp(x) for x in (_a or [])]
                    _kb = [_pair_key_from_disp(x) for x in (_b or [])]
                    _ka = [x for x in _ka if x]
                    _kb = [x for x in _kb if x]
                    return _ka == _kb
                except Exception:
                    return False

            def _weighted_car_score_map_from_flows(_summary, _value_key):
                """v223: 各流れの車番別平均評価×流れ比率を車番ごとに合算する共通関数。"""
                _ratio = _flow_ratio_map_for_trio()
                _per_car = {}
                try:
                    for _style_name, _rows in (_summary or []):
                        _w = float(_ratio.get(str(_style_name), 0.0) or 0.0)
                        for _r in (_rows or []):
                            try:
                                _car = int(_r.get("car"))
                                _v = float(_r.get(str(_value_key), 0.0) or 0.0)
                                _per_car[_car] = _per_car.get(_car, 0.0) + _v * _w
                            except Exception:
                                pass
                except Exception:
                    _per_car = {}
                return _per_car

            def _weighted_car_hit_map_from_flows():
                return _weighted_car_score_map_from_flows(flow_hit_avg_summary, "hit_avg")

            def _weighted_car_myoumi_map_from_flows():
                return _weighted_car_score_map_from_flows(flow_myoumi_avg_summary, "myoumi_avg")

            def _overall_myoumi_core_rank(_myoumi_rank):
                _r = str(_myoumi_rank)
                if _r in ("A++", "A+", "A"):
                    return "A"
                if _r in ("B", "C", "D"):
                    return _r
                return "D"

            def _overall_hit_rank_from_score(_score):
                try:
                    _s = float(_score)
                    if _s >= 10.0:
                        return "A"
                    if _s >= 8.0:
                        return "B"
                    if _s >= 6.0:
                        return "C"
                    return "D"
                except Exception:
                    return "D"

            def _overall_myoumi_rank_from_score(_score):
                try:
                    _score = float(_score)
                    if _score >= 10.0:
                        return "A++"
                    if _score >= 9.4:
                        return "A+"
                    if _score >= 8.4:
                        return "A"
                    if _score >= 7.0:
                        return "B"
                    if _score >= 5.5:
                        return "C"
                    return "D"
                except Exception:
                    return "D"

            def _overall_total_rank_from_ranks(_hit_rank, _myoumi_rank):
                _mr = _overall_myoumi_core_rank(_myoumi_rank)
                _table = {
                    ("A", "A"): "A", ("A", "B"): "A", ("A", "C"): "B", ("A", "D"): "C",
                    ("B", "A"): "A", ("B", "B"): "B", ("B", "C"): "B", ("B", "D"): "C",
                    ("C", "A"): "B", ("C", "B"): "C", ("C", "C"): "C", ("C", "D"): "D",
                    ("D", "A"): "C", ("D", "B"): "D", ("D", "C"): "D", ("D", "D"): "D",
                }
                return _table.get((str(_hit_rank), _mr), "D")

            def _overall_total_score(_hit_score, _myoumi_score):
                """
                v231:
                新しい流れ加重2車複表では、ABCDランクへ寄せず数値で見る。
                総合点は「的中点」と「妙味点」の単純平均。
                """
                try:
                    return round((max(0.0, float(_hit_score)) + max(0.0, float(_myoumi_score))) / 2.0, 1)
                except Exception:
                    return 0.0

            def _make_weighted_overall_pair_rows(_weighted_car_hit_map, _weighted_car_myoumi_map):
                """
                v231:
                全21通り2車複を、流れ加重の的中単騎評価＋妙味単騎評価から再評価する。
                ・的中点：2車の加重的中単騎評価の平均
                ・妙味点：2車の加重妙味単騎評価の平均
                ・総合点：的中点と妙味点の単純平均
                ・ABCDランクは出さず、小数点第一位の数値で表示する。
                """
                # 全通りのキーは既存評価表から取得する。欠ける場合に備え、全車からも補完する。
                _keys = set()
                try:
                    for _style_name, _rows in (flow_all_pair_pt_summary or []):
                        for _row in (_rows or []):
                            _key = _pair_key_from_disp((_row or {}).get("disp"))
                            if _key:
                                _keys.add(_key)
                except Exception:
                    pass
                try:
                    _cars = sorted(set(int(c) for c in list(_weighted_car_hit_map.keys()) + list(_weighted_car_myoumi_map.keys())))
                    for _a, _b in combinations(_cars, 2):
                        _keys.add(tuple(sorted((int(_a), int(_b)))))
                except Exception:
                    pass

                _out = []
                for _key in sorted(_keys):
                    try:
                        _a, _b = int(_key[0]), int(_key[1])
                        _ha = _weighted_car_hit_map.get(_a, None)
                        _hb = _weighted_car_hit_map.get(_b, None)
                        _ma = _weighted_car_myoumi_map.get(_a, None)
                        _mb = _weighted_car_myoumi_map.get(_b, None)
                        if _ha is None or _hb is None:
                            continue
                        if _ma is None or _mb is None:
                            continue
                        # v231: 単騎評価から2車複へ変換するため、2車の平均で同じスケールを維持する。
                        _hit_score = round(max(0.0, min(12.0, (float(_ha) + float(_hb)) / 2.0)), 2)
                        _myoumi_score = round(max(0.0, min(10.8, (float(_ma) + float(_mb)) / 2.0)), 2)
                        _hit_rank = ""
                        _myoumi_rank = ""
                        _total_rank = ""
                        _total_pt = _overall_total_score(_hit_score, _myoumi_score)
                        _out.append({
                            "disp": f"{_a}-{_b}",
                            "a": _a,
                            "b": _b,
                            "hit_score": round(float(_hit_score), 2),
                            "myoumi_score": round(float(_myoumi_score), 2),
                            "hit_rank": _hit_rank,
                            "myoumi_rank": _myoumi_rank,
                            "total_rank": _total_rank,
                            "total_pt": _total_pt,
                        })
                    except Exception:
                        pass
                try:
                    return sorted(list(_out or []), key=lambda _r: (
                        float((_r or {}).get("total_pt", 0.0) or 0.0),
                        float((_r or {}).get("hit_score", 0.0) or 0.0),
                        float((_r or {}).get("myoumi_score", 0.0) or 0.0),
                    ), reverse=True)
                except Exception:
                    return list(_out or [])

            def _select_axis3_nifuku_rows(_rows, _weighted_car_hit_map, _weighted_car_myoumi_map):
                """v223: 全通り評価から、◎軸-相手3車の2車複3点へ絞る。"""
                try:
                    _rows = list(_rows or [])
                    if not _rows:
                        return None, [], []
                    _top_rows = _rows[:10]
                    _conn_count = {}
                    _conn_pt = {}
                    for _r in _top_rows:
                        try:
                            a, b = int(_r.get("a")), int(_r.get("b"))
                            pt = float(_r.get("total_pt", 0.0) or 0.0)
                            for c, other in ((a, b), (b, a)):
                                _conn_count[c] = _conn_count.get(c, 0) + 1
                                _conn_pt[c] = _conn_pt.get(c, 0.0) + pt
                        except Exception:
                            pass

                    _hit_rows = sorted(
                        _weighted_car_hit_map.items(),
                        key=lambda kv: (float(kv[1]), float(_weighted_car_myoumi_map.get(int(kv[0]), 0.0)), int(_conn_count.get(int(kv[0]), 0)), -int(kv[0])),
                        reverse=True,
                    )
                    _axis_candidates = [int(c) for c, _ in _hit_rows[:3]] or [int(_rows[0].get("a"))]
                    _axis = max(
                        _axis_candidates,
                        key=lambda c: (
                            int(_conn_count.get(int(c), 0)),
                            float(_conn_pt.get(int(c), 0.0)),
                            float(_weighted_car_hit_map.get(int(c), 0.0)),
                            float(_weighted_car_myoumi_map.get(int(c), 0.0)),
                            -int(c),
                        )
                    )

                    _axis_rows = []
                    for _r in _rows:
                        try:
                            a, b = int(_r.get("a")), int(_r.get("b"))
                            if int(_axis) in (a, b):
                                _axis_rows.append(_r)
                        except Exception:
                            pass
                    _axis_rows = sorted(_axis_rows, key=lambda _r: (
                        float((_r or {}).get("total_pt", 0.0) or 0.0),
                        float((_r or {}).get("hit_score", 0.0) or 0.0),
                        float((_r or {}).get("myoumi_score", 0.0) or 0.0),
                    ), reverse=True)
                    _main = _axis_rows[:3]
                    return int(_axis), _main, _axis_rows
                except Exception:
                    return None, [], []


            _weighted_car_hit_map = _weighted_car_hit_map_from_flows()
            _weighted_car_myoumi_map = _weighted_car_myoumi_map_from_flows()
            _nifuku_axis = None
            _nifuku_axis_rows_all = []

            for _style_name, _rows in (flow_buy_pt_summary or []):
                _summary_pt_map[str(_style_name)] = list(_rows or [])

            # v204/v205:
            # 全体の本線/抑えは、各流れの「上位2点」ではなく、
            # 総合B以上候補全体を重複除外して作る。
            # 同じ買目が複数流れに出た場合、表示ptと妙味期待は最もptが高い流れの値を採用する。
            for _style_name, _rows in (flow_b_candidate_pt_summary or []):
                for _row in (_rows or []):
                    try:
                        _key = _pair_key_from_disp(_row.get("disp"))
                        if not _key:
                            continue
                        _disp = f"{_key[0]}-{_key[1]}"
                        _pt = float(_row.get("total_pt", 0.0) or 0.0)
                        _myoumi_rank = str(_row.get("myoumi_rank", "") or "")
                        if _key in _overall_seen:
                            for _old in _overall_pair_rows:
                                try:
                                    if _pair_key_from_disp(_old.get("disp")) == _key and _pt > float(_old.get("total_pt", 0.0) or 0.0):
                                        _old["total_pt"] = _pt
                                        _old["myoumi_rank"] = _myoumi_rank
                                except Exception:
                                    pass
                            continue
                        _overall_seen.add(_key)
                        _overall_pairs.append(_disp)
                        _overall_pair_rows.append({"disp": _disp, "total_pt": _pt, "myoumi_rank": _myoumi_rank})
                    except Exception:
                        pass

            # v220: 2車複サマリーは、流れ別候補の最大ptではなく、
            #       流れ配分込みの車番別平均評価で的中期待を再計算した全通り評価から作る。
            _weighted_all_pair_rows = _make_weighted_overall_pair_rows(_weighted_car_hit_map, _weighted_car_myoumi_map)
            if _weighted_all_pair_rows:
                # v225:
                # 2車複は軸を先に決めない。
                # 流れ加重的中単騎評価＋流れ加重妙味単騎評価から作った
                # 全21通りの加重2車複評価表そのものを母集団にし、
                # 最終本線は総合pt上位3点を採用する。
                _nifuku_axis = None
                _nifuku_axis_rows_all = []
                _overall_pair_rows = list(_weighted_all_pair_rows or [])
                _overall_pairs = [str(_r.get("disp")) for _r in (_overall_pair_rows or []) if _r.get("disp")]
                _overall_seen = {_pair_key_from_disp(_p) for _p in (_overall_pairs or []) if _pair_key_from_disp(_p)}

            # 流れ別表示用には総合B以上候補の車券名だけを保持する。
            for _style_name, _pairs in (flow_b_candidate_summary or []):
                _b_candidate_map[str(_style_name)] = list(_pairs or [])

            _candidate_pair_styles = {}
            _candidate_pair_order = []
            _candidate_pair_best_row = {}
            for _style_name, _rows in (flow_b_candidate_pt_summary or []):
                for _row in (_rows or []):
                    _key = _pair_key_from_disp((_row or {}).get("disp"))
                    if not _key:
                        continue
                    if _key not in _candidate_pair_styles:
                        _candidate_pair_styles[_key] = []
                        _candidate_pair_order.append(_key)
                    if str(_style_name) not in _candidate_pair_styles[_key]:
                        _candidate_pair_styles[_key].append(str(_style_name))
                    try:
                        _pt = float((_row or {}).get("total_pt", 0.0) or 0.0)
                        _old = _candidate_pair_best_row.get(_key)
                        if _old is None or _pt > float((_old or {}).get("total_pt", 0.0) or 0.0):
                            _candidate_pair_best_row[_key] = {
                                "disp": f"{_key[0]}-{_key[1]}",
                                "total_pt": _pt,
                                "myoumi_rank": str((_row or {}).get("myoumi_rank", "") or ""),
                            }
                    except Exception:
                        pass

            # v210: イチオシの判定前に、本線/抑えの表示対象キーを先に決める。
            def _sort_rows_by_pt_desc(_rows):
                try:
                    return sorted(list(_rows or []), key=lambda _r: float((_r or {}).get("total_pt", 0.0) or 0.0), reverse=True)
                except Exception:
                    return list(_rows or [])

            _overall_sorted_rows = _sort_rows_by_pt_desc(_overall_pair_rows)
            try:
                import math as _math
                _n_all = len(_overall_sorted_rows or [])
                _main_n = max(1, int(_math.ceil(_n_all * _NIFUKU_MAIN_PERCENT))) if _n_all > 0 else 0
                _main_n = min(_main_n, int(_NIFUKU_MAIN_MAX)) if _main_n > 0 else 0
                _display_n = max(_main_n, int(_math.ceil(_n_all * _NIFUKU_DISPLAY_PERCENT))) if _n_all > 0 else 0
                _display_n = min(_display_n, int(_NIFUKU_DISPLAY_MAX)) if _display_n > 0 else 0
                _display_n = max(_display_n, _main_n)
            except Exception:
                _n_all = len(_overall_sorted_rows or [])
                _main_n = min(int(_NIFUKU_MAIN_MAX), _n_all)
                _display_n = min(int(_NIFUKU_DISPLAY_MAX), _n_all)
            _overall_main_rows = list(_overall_sorted_rows[:_main_n])
            _overall_sub_rows = list(_overall_sorted_rows[_main_n:_display_n])
            _display_pair_key_set = set()
            for _r in list(_overall_main_rows) + list(_overall_sub_rows):
                try:
                    _k = _pair_key_from_disp((_r or {}).get("disp"))
                    if _k:
                        _display_pair_key_set.add(_k)
                except Exception:
                    pass

            # v211: 「イチオシ」は廃止。
            #       代わりに、各流れの総合B以上候補・総合pt上位10内で
            #       複数流れに重複した買目を「ベスト10内重複」として表示する。
            _best10_pair_styles = {}
            _best10_pair_order = []
            _best10_pair_best_row = {}
            for _style_name, _rows in (flow_b_candidate_pt_summary or []):
                try:
                    _top10_rows = _sort_rows_by_pt_desc(_rows or [])[:10]
                except Exception:
                    _top10_rows = list(_rows or [])[:10]
                for _row in (_top10_rows or []):
                    _key = _pair_key_from_disp((_row or {}).get("disp"))
                    if not _key:
                        continue
                    if _key not in _best10_pair_styles:
                        _best10_pair_styles[_key] = []
                        _best10_pair_order.append(_key)
                    if str(_style_name) not in _best10_pair_styles[_key]:
                        _best10_pair_styles[_key].append(str(_style_name))
                    try:
                        _pt = float((_row or {}).get("total_pt", 0.0) or 0.0)
                        _old = _best10_pair_best_row.get(_key)
                        if _old is None or _pt > float((_old or {}).get("total_pt", 0.0) or 0.0):
                            _best10_pair_best_row[_key] = {
                                "disp": f"{_key[0]}-{_key[1]}",
                                "total_pt": _pt,
                                "myoumi_rank": str((_row or {}).get("myoumi_rank", "") or ""),
                            }
                    except Exception:
                        pass

            _best10_overlap_parts = []
            for _key in _best10_pair_order:
                _styles = _best10_pair_styles.get(_key, []) or []
                if len(_styles) >= 2:
                    _best = _best10_pair_best_row.get(_key, {}) or {}
                    try:
                        _best_pt = float(_best.get("total_pt", 0.0) or 0.0)
                    except Exception:
                        _best_pt = 0.0
                    _pt_txt = f"／{_best_pt:.1f}"
                    _myoumi_rank = str(_best.get("myoumi_rank", "") or "").strip()
                    _myoumi_txt = f" 妙味期待{_myoumi_rank}" if _myoumi_rank else ""
                    _best10_overlap_parts.append(f"{_key[0]}-{_key[1]}（{'・'.join(_styles)}{_pt_txt}{_myoumi_txt}）")

            def _fmt_overall_rows_with_pt(_rows, include_myoumi=False):
                _parts = []
                for _r in (_rows or []):
                    try:
                        _disp = str(_r.get("disp", "")).strip()
                        if not _disp:
                            continue
                        _pt = float(_r.get("total_pt", 0.0) or 0.0)
                        if include_myoumi:
                            _myoumi_rank = str(_r.get("myoumi_rank", "") or "").strip()
                            _myoumi_txt = f" 妙味期待{_myoumi_rank}" if _myoumi_rank else ""
                            _parts.append(f"{_disp}（{_pt:.1f}{_myoumi_txt}）")
                        else:
                            _parts.append(f"{_disp}（{_pt:.1f}）")
                    except Exception:
                        pass
                return "　".join(_parts) if _parts else "該当なし"

            def _sort_rows_by_pt_desc(_rows):
                try:
                    return sorted(list(_rows or []), key=lambda _r: float((_r or {}).get("total_pt", 0.0) or 0.0), reverse=True)
                except Exception:
                    return list(_rows or [])

            # v225: 最終2車複は◎軸流しではなく、加重2車複評価表の総合pt上位3点。
            _overall_sorted_rows = _sort_rows_by_pt_desc(_overall_pair_rows)
            _overall_main_rows = list(_overall_sorted_rows[:3])
            _overall_sub_rows = []
            _display_pair_key_set = set()
            for _r in list(_overall_main_rows) + list(_overall_sub_rows):
                try:
                    _k = _pair_key_from_disp((_r or {}).get("disp"))
                    if _k:
                        _display_pair_key_set.add(_k)
                except Exception:
                    pass

            def _make_flow_weighted_trio_lines():
                """
                v220:
                2車複サマリーにも使った流れ配分込みの車番別平均評価を、
                そのまま3連複 A-BCD-BCD の軸・ヒモ決定にも使う。
                """
                try:
                    _per_car = dict(_weighted_car_hit_map or {})
                    _per_myoumi = dict(_weighted_car_myoumi_map or {})
                    if not _per_car:
                        return []
                    _weighted_rows = sorted(_per_car.items(), key=lambda kv: (float(kv[1]), -int(kv[0])), reverse=True)
                    _weighted_line = " → ".join(f"{int(c)}（{float(v):.1f}）" for c, v in _weighted_rows)
                    _myoumi_rows = sorted(_per_myoumi.items(), key=lambda kv: (float(kv[1]), -int(kv[0])), reverse=True)
                    _myoumi_line = " → ".join(f"{int(c)}（{float(v):.1f}）" for c, v in _myoumi_rows)

                    # 表示対象2車複（本線＋抑え）との接続。
                    _display_rows = list(_overall_main_rows or []) + list(_overall_sub_rows or [])
                    _conn = {}
                    _main_conn = {}
                    _overlap_conn = {}
                    for _r in (_display_rows or []):
                        _k = _pair_key_from_disp((_r or {}).get("disp"))
                        if not _k:
                            continue
                        a, b = int(_k[0]), int(_k[1])
                        _conn.setdefault(a, set()).add(b)
                        _conn.setdefault(b, set()).add(a)
                    for _r in (_overall_main_rows or []):
                        _k = _pair_key_from_disp((_r or {}).get("disp"))
                        if not _k:
                            continue
                        a, b = int(_k[0]), int(_k[1])
                        _main_conn.setdefault(a, set()).add(b)
                        _main_conn.setdefault(b, set()).add(a)
                    for _key in (_best10_pair_order or []):
                        try:
                            _styles = _best10_pair_styles.get(_key, []) or []
                            if len(_styles) >= 2:
                                a, b = int(_key[0]), int(_key[1])
                                _overlap_conn.setdefault(a, set()).add(b)
                                _overlap_conn.setdefault(b, set()).add(a)
                        except Exception:
                            pass

                    _top_for_axis = [int(c) for c, _ in _weighted_rows[:3]]
                    if not _top_for_axis:
                        _base = [f"流れ加重的中単騎評価】{_weighted_line}"]
                        if _myoumi_line:
                            _base.append(f"流れ加重妙味単騎評価】{_myoumi_line}")
                        return _base
                    _axis = int(_nifuku_axis) if _nifuku_axis is not None else max(
                        _top_for_axis,
                        key=lambda c: (len(_conn.get(int(c), set())), float(_per_car.get(int(c), 0.0)), -int(c))
                    )

                    def _add_unique(_lst, _x):
                        try:
                            _x = int(_x)
                            if _x != int(_axis) and _x not in _lst:
                                _lst.append(_x)
                        except Exception:
                            pass

                    # v240 本線ヒモ：
                    # 三連複は「軸-ヒモ-ヒモ」の3点フォーメーションで作る。
                    # ヒモ3車は、まず軸が含まれる実ラインの相手を優先し、
                    # 不足分を「軸と組み合わせた加重2車複総合点上位」で補完する。
                    # 例：軸7・ライン726・軸絡み総合上位5-7なら 7-265-265。
                    def _axis_line_mates(_axis_no):
                        _mates = []
                        try:
                            _axis_no = int(_axis_no)
                            _line_sources = []
                            try:
                                _line_sources.append(globals().get("lines_live", None))
                            except Exception:
                                pass
                            try:
                                _line_sources.append(lines_live)
                            except Exception:
                                pass
                            try:
                                _line_def = globals().get("line_def_live", None)
                                if isinstance(_line_def, dict):
                                    _line_sources.append(list(_line_def.values()))
                            except Exception:
                                pass
                            for _lines in (_line_sources or []):
                                if not _lines:
                                    continue
                                for _ln in (_lines or []):
                                    try:
                                        _cars = [int(x) for x in (_ln or []) if str(x).isdigit()]
                                    except Exception:
                                        _cars = []
                                    if _axis_no not in _cars:
                                        continue
                                    for _c in _cars:
                                        if _c != _axis_no and _c not in _mates:
                                            _mates.append(_c)
                                    if _mates:
                                        return _mates
                        except Exception:
                            pass
                        return _mates

                    def _axis_pair_score_partners(_axis_no):
                        _partners = []
                        try:
                            _axis_no = int(_axis_no)
                            _rows = sorted(list(_overall_pair_rows or []), key=lambda _r: (
                                float((_r or {}).get("total_pt", 0.0) or 0.0),
                                float((_r or {}).get("hit_score", 0.0) or 0.0),
                                float((_r or {}).get("myoumi_score", 0.0) or 0.0),
                            ), reverse=True)
                            for _r in (_rows or []):
                                try:
                                    a, b = int(_r.get("a")), int(_r.get("b"))
                                    if _axis_no not in (a, b):
                                        continue
                                    _p = b if a == _axis_no else a
                                    if _p != _axis_no and _p not in _partners:
                                        _partners.append(_p)
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        return _partners

                    _main_cols = []
                    for _c in _axis_line_mates(_axis):
                        _add_unique(_main_cols, _c)
                    for _c in _axis_pair_score_partners(_axis):
                        _add_unique(_main_cols, _c)
                    # 保険：ライン・軸絡み2車複で不足する場合だけ、加重的中単騎順で補完する。
                    for c, _v in _weighted_rows:
                        _add_unique(_main_cols, c)
                    _main_cols = _main_cols[:3]

                    # 広め列：本線列に、加重単騎評価の次点を1車追加（最大4車）。
                    # 追加車は、最後の保護相手（例：ベスト10内重複相手）の手前へ入れる。
                    # 例：本線 5-627-627 → 広め 5-6237-6237。
                    _wide_cols = list(_main_cols)
                    for c, _v in _weighted_rows:
                        try:
                            c = int(c)
                            if c == int(_axis) or c in _wide_cols:
                                continue
                            if len(_wide_cols) >= 3:
                                _wide_cols.insert(len(_wide_cols) - 1, c)
                            else:
                                _wide_cols.append(c)
                            break
                        except Exception:
                            pass
                    _wide_cols = _wide_cols[:4]

                    _out = [f"流れ加重的中単騎評価】{_weighted_line}"]
                    if _myoumi_line:
                        _out.append(f"流れ加重妙味単騎評価】{_myoumi_line}")
                    if len(_main_cols) >= 2:
                        _out.append("流れ加重3連複】")
                        _out.append(f"本線】{_fmt_trio_form(_axis, _main_cols)}（{_trio_form_ticket_count(_main_cols)}点）")
                        if len(_wide_cols) > len(_main_cols):
                            _out.append(f"広め】{_fmt_trio_form(_axis, _wide_cols)}（{_trio_form_ticket_count(_wide_cols)}点）")
                    return _out
                except Exception:
                    return []

            lines.append(_fmt_flow_ratio_line(_flow_ratio_map_for_trio()))
            lines.append("")

            # v231:
            # 加重2車複評価表はABCDを出さず、的中点・妙味点・総合点を小数点第一位で表示する。
            def _fmt_weighted_pair_table(_rows, _limit=21):
                """
                v237:
                旧ヴェロビ表に近い全角スペース主体の整形。
                罫線なし。半角スペースの大量挿入は使わない。
                """
                try:
                    _rows = list(_rows or [])[:int(_limit)]
                    if not _rows:
                        return ["該当なし"]

                    def _fmt_num(_v):
                        try:
                            return f"{float(_v):.1f}"
                        except Exception:
                            return "-"

                    def _cell_left(_text, _chars):
                        _s = str(_text)
                        return _s + ("　" * max(0, int(_chars) - len(_s)))

                    def _cell_right(_text, _chars):
                        _s = str(_text)
                        return ("　" * max(0, int(_chars) - len(_s))) + _s

                    # 旧表の見た目に寄せる。
                    # 買い目は3桁固定、数値は小数1桁の3桁固定。
                    _out = []
                    _out.append("　買い目　　 的中点　 妙味点　 総合点")
                    for _r in _rows:
                        try:
                            _disp = str(_r.get("disp", "")).strip()
                            if not _disp:
                                continue
                            _hit = _fmt_num(_r.get("hit_score", 0.0))
                            _myo = _fmt_num(_r.get("myoumi_score", 0.0))
                            _tot = _fmt_num(_r.get("total_pt", 0.0))
                            _out.append(
                                "　" + _cell_left(_disp, 3) + "　　　　" +
                                _cell_right(_hit, 3) + "　　　" +
                                _cell_right(_myo, 3) + "　　　" +
                                _cell_right(_tot, 3)
                            )
                        except Exception:
                            pass
                    return _out if _out else ["該当なし"]
                except Exception:
                    return ["該当なし"]

            _fw_trio_lines = _make_flow_weighted_trio_lines()

            # v227: note上部は買い目主役で最小限にする。
            # 詳細な加重2車複評価表・買い目根拠・流れ別買目考察は出さない。
            lines.append("【買い目サマリー】")
            # v238: 優先券種を三連複 → 2車複へ変更。
            # 全体妙味の率も3連複軸を指すため、先に3連複を表示する。
            if _fw_trio_lines:
                for _ln in _fw_trio_lines:
                    _s = str(_ln)
                    if _s.startswith("流れ加重3連複】"):
                        lines.append("3連複 本線】")
                    elif _s.startswith("本線】") or _s.startswith("広め】"):
                        lines.append(_s)
            lines.append(f"2車複 本線3点】{_fmt_overall_rows_with_pt(_overall_main_rows, include_myoumi=False)}")
            lines.append("")
            lines.append("")

            # v227: 検証に必要な総合加重単騎評価だけ残す。
            if _fw_trio_lines:
                _score_lines = [
                    str(x) for x in _fw_trio_lines
                    if str(x).startswith("流れ加重的中単騎評価】")
                    or str(x).startswith("流れ加重妙味単騎評価】")
                ]
                if _score_lines:
                    lines.append("【総合加重単騎評価】")
                    lines.extend(_score_lines)
                    lines.append("")
                    lines.append("【加重2車複評価表】")
                    lines.extend(_fmt_weighted_pair_table(_overall_sorted_rows, _limit=21))
                    lines.append("")
                    lines.append("")
        else:
            lines.append("【買目考察】")
            lines.append("")
            lines.append("生成不可")
            lines.append("")
        # v228: 意味が伝わらない「コピー用：xxxx」は表示しない。
        return "\n".join(lines).strip()
    except Exception as e:
        return f"note最終推奨サマリー生成不可：{e}"



try:
    _rec_style = globals().get("RECOMMENDED_STYLE", "")
    _rec_seq = globals().get("RECOMMENDED_STYLE_SEQ", [])
    _rec_copy = globals().get("RECOMMENDED_STYLE_COPY", "")
    _rec_seq = [int(x) for x in (_rec_seq or []) if str(x).isdigit()]

    # まず軸評価行を全体妙味へ置換（この時点では旧ラベルのまま）
    note_text = _replace_axis_line_to_expect(note_text, expect_axis_label)

    # 既存の上部サマリーだけを削除
    note_text = _strip_existing_top_summary(note_text)

    summary_block = "\n\n" + _make_note_final_summary_block(
        _rec_style,
        _rec_seq,
        _rec_copy,
        expect_axis_label,
        rule_buy_block,
        market_mark_map,
    ) + "\n"

    # 最初の全体妙味行の直後にだけ挿入
    _m_axis = re.search(
        r"全体妙味：(?:AA|A|B|C|荒|低)（(?:三連複軸想定着内率|軸想定2着内率)\s*\d+%）",
        note_text
    )

    if _m_axis:
        note_text = note_text.replace(
            _m_axis.group(0),
            _m_axis.group(0) + summary_block,
            1
        )
    else:
        note_text = summary_block + "\n\n" + note_text

    note_text = _display_expect_myoumi_labels_in_text(note_text)

except Exception as _e:
    st.caption(f"note上部サマリー生成不可：{_e}")



# -----------------------------------------
# noteコピー表示整理（表示だけ。計算・順位・フォメ生成には触らない）
# 削除対象：
# ・ラスト半周補正ブロック
# ・会場×最終Hライン補正ブロック
# ・下部の重複した推奨戦法〜コピー用
# ・軸評価〜2車複候補〜絞り推奨買目〜仮想単勝
# 残す対象：
# ・上部サマリー
# ・ライン評価グループ
# ・KO使用スコア
# ・順流/渦/逆流メイン着順予想
# ・短評
# -----------------------------------------
def _clean_note_copy_display_only(text: str) -> str:
    try:
        lines = str(text).splitlines()
        out = []
        i = 0
        n = len(lines)

        while i < n:
            line = lines[i]
            s = line.strip()

            # v228: 意味不明なコピー用行は全削除
            if s.startswith("コピー用："):
                i += 1
                while i < n and lines[i].strip() == "":
                    i += 1
                continue

            # 1) ラスト半周補正ブロックを削除
            if s == "【ラスト半周補正】":
                i += 1
                # 次の空行まで飛ばす
                while i < n and lines[i].strip() != "":
                    i += 1
                # 空行も1つ飛ばす
                while i < n and lines[i].strip() == "":
                    i += 1
                continue

            # 1-b) 会場×最終Hライン補正ブロックを削除
            if s == "【会場×最終Hライン補正】":
                i += 1
                # 次の空行まで飛ばす
                while i < n and lines[i].strip() != "":
                    i += 1
                # 空行も1つ飛ばす
                while i < n and lines[i].strip() == "":
                    i += 1
                continue

            # 2) 下部の重複した推奨戦法〜仮想単勝まで削除
            #    戦法別着順予想の後に出る plain な「推奨戦法：」から始まるブロックだけ対象。
            if re.match(r"^推奨戦法：", s):
                i += 1
                # ＜短評＞直前まで飛ばす
                while i < n and lines[i].strip() != "＜短評＞":
                    i += 1
                # ＜短評＞は残すので continue せず、次ループで処理
                continue

            # 3) 念のため、軸評価から始まってしまった下部買い目ブロックも削除
            if re.match(r"^軸評価：", s):
                i += 1
                while i < n and lines[i].strip() != "＜短評＞":
                    i += 1
                continue

            out.append(line)
            i += 1

        # 連続空行を最大2行に抑える
        cleaned = []
        blank = 0
        for line in out:
            if line.strip() == "":
                blank += 1
                if blank <= 2:
                    cleaned.append(line)
            else:
                blank = 0
                cleaned.append(line)

        return "\n".join(cleaned).strip() + "\n"

    except Exception:
        return text



# -----------------------------------------
# 短評をアプリ向け定型コメントへ置換（表示だけ。計算・順位・フォメ生成には触らない）
# -----------------------------------------
def _replace_tanpyou_with_simple_comment(text: str) -> str:
    try:
        txt = str(text)

        # 全体妙味を取得
        m_myoumi = re.search(r"全体妙味：([^\s（]+)", txt)
        myoumi = m_myoumi.group(1).strip() if m_myoumi else "未判定"

        # 順当度を旧短評から取得
        m_jundo = re.search(r"・順当度：([^［\n]+)", txt)
        jundo = m_jundo.group(1).strip() if m_jundo else "未判定"

        # 推奨戦法を取得
        m_style = re.search(r"✅\s*推奨戦法：([^\n]+)", txt)
        style = m_style.group(1).strip() if m_style else "推奨戦法"

        # 全体妙味コメント
        # note_text側で旧ラベル→新A/B/C変換済み。ここで再変換するとAがBへズレるため行わない。
        myoumi = str(myoumi or "").strip()
        if myoumi == "A":
            line1 = "・全体妙味：A。市場評価と近い構成。"
        elif myoumi == "B":
            line1 = "・全体妙味：B。市場評価とのズレは中間。"
        elif myoumi == "C":
            line1 = "・全体妙味：C。妙味は控えめ。"
        else:
            line1 = "・全体妙味：未判定。"

        # 展開コメント
        if jundo and jundo != "未判定":
            line2 = f"・展開は{jundo}。"
        else:
            line2 = "・展開は未判定。"

        # 戦法コメント
        if style in ("順流", "渦", "逆流"):
            line3 = f"・{style}メインで確認。"
        else:
            line3 = "・推奨戦法を中心に確認。"

        new_block = "＜短評＞\n" + "\n".join([line1, line2, line3])

        # ＜短評＞以降を定型コメントに差し替える
        if "＜短評＞" in txt:
            txt = re.sub(r"＜短評＞[\s\S]*$", new_block + "\n", txt)
        else:
            txt = txt.rstrip() + "\n\n" + new_block + "\n"

        return txt
    except Exception:
        return text

note_text = _clean_note_copy_display_only(note_text)
note_text = _replace_tanpyou_with_simple_comment(note_text)

st.text_area("ここを選択してコピー", note_text, height=620)
# =========================


# =========================
#  一括置換ブロック ここまで
# =========================
