#pragma once
#include <algorithm>
#include <cmath>
#include "constants.h"

// ── Card features (4-float encoding) ─────────────────────────────────────────

inline void card_features(int card, float* out) {
    if (card < 0) { out[0]=out[1]=out[2]=out[3]=0.f; return; }
    out[0] = (float)card_rank(card) / (NUM_RANKS - 1);
    int s = card_suit(card);
    out[1]=(s==0)?1.f:0.f; out[2]=(s==1)?1.f:0.f; out[3]=(s==2)?1.f:0.f;
}

// ── Hand strength features (6) ────────────────────────────────────────────────

inline void hand_strength_features(const int* hand2, const int* community,
                                    int n_comm, float* out) {
    if (hand2[0] < 0 || hand2[1] < 0) { for(int i=0;i<6;i++) out[i]=0.f; return; }
    int r0=card_rank(hand2[0]),r1=card_rank(hand2[1]);
    int s0=card_suit(hand2[0]),s1=card_suit(hand2[1]);
    out[0]=(float)std::max(r0,r1)/(NUM_RANKS-1);
    out[1]=(r0==r1)?1.f:0.f; out[2]=(s0==s1)?1.f:0.f; out[3]=0.f;
    if (n_comm > 0) {
        int sc[3]={}; sc[s0]++; sc[s1]++;
        for(int i=0;i<n_comm;i++) if(community[i]>=0) sc[card_suit(community[i])]++;
        int mx=*std::max_element(sc,sc+3);
        out[3]=(mx>=4)?1.f:(mx>=3)?0.5f:0.f;
    }
    int gap=std::abs(r0-r1);
    out[4]=(gap<=1)?1.f:(gap<=3)?0.5f:0.f;
    out[5]=0.f;
    if (n_comm > 0) {
        bool hit0=false,hit1=false; int mb=-1;
        for(int i=0;i<n_comm;i++) if(community[i]>=0) {
            int br=card_rank(community[i]);
            if(br==r0) hit0=true; if(br==r1) hit1=true;
            mb=std::max(mb,br);
        }
        if(hit0&&hit1) out[5]=1.f;
        else if(hit0||hit1) out[5]=(std::max(r0,r1)==mb)?0.75f:0.5f;
    } else {
        out[4]=(gap<=1)?1.f:(gap<=3)?0.5f:0.f;
    }
}

// ── Opponent range estimate features (6) ─────────────────────────────────────

inline void opp_range_features(const int* opp_disc, const int* community,
                                int n_comm, float* out) {
    bool has_disc=false;
    for(int i=0;i<3;i++) if(opp_disc[i]>=0) has_disc=true;
    if(!has_disc){ out[0]=out[1]=out[2]=0.5f; out[3]=out[4]=out[5]=0.f; return; }
    int dr[3],ds[3]; int nd=0;
    for(int i=0;i<3;i++) if(opp_disc[i]>=0){ dr[nd]=card_rank(opp_disc[i]); ds[nd]=card_suit(opp_disc[i]); nd++; }
    float sum_r=0; for(int i=0;i<nd;i++) sum_r+=dr[i];
    float avg=sum_r/nd/(NUM_RANKS-1);
    out[0]=avg; out[1]=(float)*std::max_element(dr,dr+nd)/(NUM_RANKS-1);
    bool hp=false;
    for(int i=0;i<nd&&!hp;i++) for(int j=i+1;j<nd;j++) if(dr[i]==dr[j]){hp=true;break;}
    out[2]=hp?1.f:0.f;
    int sc[3]={}; for(int i=0;i<nd;i++) sc[ds[i]]++;
    out[3]=(float)*std::max_element(sc,sc+3)/3.f;
    out[4]=0.f;
    if(n_comm>0){
        int bs[3]={}; for(int i=0;i<n_comm;i++) if(community[i]>=0) bs[card_suit(community[i])]++;
        int dom=(int)(std::max_element(bs,bs+3)-bs);
        int match=0; for(int i=0;i<nd;i++) if(ds[i]==dom) match++;
        out[4]=(float)match/3.f;
    }
    out[5]=1.f-avg;
}

// ── Full 119-dim feature vector ───────────────────────────────────────────────

inline void state_to_features(
    const int* hero_hand2, const int* hero_hand5,
    const int* community, int n_comm,
    int my_bet, int opp_bet, int street, bool is_bb,
    const int* my_disc, const int* opp_disc,
    bool use_hand5,
    float* features,
    const int street_bets[4][2]        = nullptr,
    const float* street_last_ratios    = nullptr,
    const int*   street_bet_counts     = nullptr,
    const int*   history_players       = nullptr,
    const int*   history_actions       = nullptr,
    int          history_len           = 0,
    int          num_acts_this_street  = 0
) {
    for(int i=0;i<FEATURE_DIM;i++) features[i]=0.f;
    int idx=0;

    // [0-19] Hero hand
    if(use_hand5&&hero_hand5){
        int h5[5]; std::copy(hero_hand5,hero_hand5+5,h5); std::sort(h5,h5+5);
        for(int i=0;i<5;i++) card_features(h5[i],&features[idx+i*4]);
    } else {
        int h2[2]={hero_hand2[0],hero_hand2[1]};
        if(h2[0]>h2[1]) std::swap(h2[0],h2[1]);
        card_features(h2[0],&features[idx]); card_features(h2[1],&features[idx+4]);
        for(int i=8;i<20;i++) features[idx+i]=0.f;
    }
    idx+=20;

    // [20-39] Community (flop sorted)
    int sc[5]={-1,-1,-1,-1,-1};
    if(n_comm>=3){
        int flop[3]={community[0],community[1],community[2]}; std::sort(flop,flop+3);
        sc[0]=flop[0]; sc[1]=flop[1]; sc[2]=flop[2];
        if(n_comm>=4) sc[3]=community[3]; if(n_comm>=5) sc[4]=community[4];
    } else for(int i=0;i<n_comm;i++) sc[i]=community[i];
    for(int i=0;i<5;i++){
        if(i<n_comm&&sc[i]>=0) card_features(sc[i],&features[idx+i*4]);
        else for(int j=0;j<4;j++) features[idx+i*4+j]=0.f;
    }
    idx+=20;

    // [40-51] My discards, [52-63] Opp discards
    for(int i=0;i<3;i++){ if(my_disc&&my_disc[i]>=0) card_features(my_disc[i],&features[idx+i*4]); else for(int j=0;j<4;j++) features[idx+i*4+j]=0.f; }
    idx+=12;
    for(int i=0;i<3;i++){ if(opp_disc&&opp_disc[i]>=0) card_features(opp_disc[i],&features[idx+i*4]); else for(int j=0;j<4;j++) features[idx+i*4+j]=0.f; }
    idx+=12;

    // [64-67] Street one-hot, [68] position
    for(int s=0;s<4;s++) features[idx++]=(street==s)?1.f:0.f;
    features[idx++]=is_bb?1.f:0.f;

    // [69-72] Bet info absolute
    int pot=my_bet+opp_bet;
    features[idx++]=(float)my_bet/MAX_BET; features[idx++]=(float)opp_bet/MAX_BET;
    features[idx++]=(float)pot/(2*MAX_BET); features[idx++]=(float)std::max(opp_bet-my_bet,0)/MAX_BET;

    // [73-78] Hand strength
    int vis_comm[5]; int vis_n=std::min(n_comm,(street==0)?0:(street==1)?3:(street==2)?4:5);
    for(int i=0;i<vis_n;i++) vis_comm[i]=community[i];
    if(street>0&&hero_hand2[0]>=0){
        hand_strength_features(hero_hand2,vis_comm,vis_n,&features[idx]);
    } else if(use_hand5&&hero_hand5){
        int ranks[5]; for(int i=0;i<5;i++) ranks[i]=card_rank(hero_hand5[i]);
        std::sort(ranks,ranks+5,std::greater<int>());
        features[idx]=(float)ranks[0]/(NUM_RANKS-1);
        bool hp2=false; for(int i=0;i<5&&!hp2;i++) for(int j=i+1;j<5;j++) if(ranks[i]==ranks[j]){hp2=true;break;}
        features[idx+1]=hp2?1.f:0.f;
        int suits[5]; for(int i=0;i<5;i++) suits[i]=card_suit(hero_hand5[i]);
        int msc=0; for(int s2=0;s2<3;s2++){int cnt=0;for(int i=0;i<5;i++) if(suits[i]==s2) cnt++;msc=std::max(msc,cnt);}
        features[idx+2]=(float)msc/5.f; features[idx+3]=features[idx+4]=features[idx+5]=0.f;
    } else { for(int i=0;i<6;i++) features[idx+i]=0.5f; }
    idx+=6;

    // [79-86] Betting history (last bet/pot ratio)
    int hp=is_bb?1:0;
    for(int s=0;s<4;s++){
        float my_r=0.f,op_r=0.f;
        if(street_last_ratios){ my_r=std::min(street_last_ratios[s*2+hp],4.f); op_r=std::min(street_last_ratios[s*2+1-hp],4.f); }
        else if(street_bets){ my_r=std::min((float)street_bets[s][hp]/(float)MAX_BET,1.f); op_r=std::min((float)street_bets[s][1-hp]/(float)MAX_BET,1.f); }
        features[idx++]=my_r; features[idx++]=op_r;
    }

    // [87-92] Opp range
    int opp_d[3]={-1,-1,-1};
    if(opp_disc){opp_d[0]=opp_disc[0];opp_d[1]=opp_disc[1];opp_d[2]=opp_disc[2];}
    opp_range_features(opp_d,vis_comm,vis_n,&features[idx]); idx+=6;
    // idx==93

    // ── Extra 26 dims ─────────────────────────────────────────────────────
    int to_call=std::max(opp_bet-my_bet,0);

    // [93-94] Initiative
    if(history_players&&history_actions&&history_len>0){
        for(int i=history_len-1;i>=0;i--){ if(history_actions[i]>=3){ features[idx]=(history_players[i]==hp)?1.f:0.f; features[idx+1]=(history_players[i]!=hp)?1.f:0.f; break; } }
    }
    idx+=2;

    // [95-96] Action context
    features[idx++]=(to_call>0)?1.f:0.f; features[idx++]=(to_call==0)?1.f:0.f;

    // [97-100] Line class
    { int bets_this=0,start=history_len-num_acts_this_street; if(start<0) start=0;
      if(history_actions) for(int i=start;i<history_len;i++) if(history_actions[i]>=3) bets_this++;
      if(to_call==0&&bets_this==0) features[idx]=1.f;
      else if(to_call>0&&bets_this==1) features[idx+1]=1.f;
      else if(to_call>0&&bets_this>=2) features[idx+2]=1.f;
      else if(to_call==0&&bets_this>=1) features[idx+3]=1.f;
      idx+=4; }

    // [101-105] Board texture
    { int br[5],bs[5],bsc[3]={}; bool paired=false,seen[9]={}; int min_r2=8,max_r2=0;
      for(int i=0;i<n_comm;i++){ int c=community[i]; if(c<0) continue; int r=card_rank(c),s2=card_suit(c); br[i]=r;bs[i]=s2; bsc[s2]++; if(seen[r]) paired=true; seen[r]=true; min_r2=std::min(min_r2,r);max_r2=std::max(max_r2,r); }
      int msc2=*std::max_element(bsc,bsc+3);
      features[idx]=paired?1.f:0.f;
      features[idx+1]=(n_comm>=3&&msc2==n_comm)?1.f:0.f;
      features[idx+2]=(msc2>=2)?1.f:0.f;
      if(n_comm>=3) features[idx+3]=((max_r2-min_r2)<=4)?1.f:0.f;
      if(n_comm>=4){ int ps[3]={}; for(int i=0;i<n_comm-1;i++) ps[bs[i]]++; if(ps[bs[n_comm-1]]>=2) features[idx+4]=1.f; }
      idx+=5; }

    // [106-110] Bet ratios
    { float sp=std::max((float)pot,1.f); int mr2=std::max(MAX_BET-std::max(my_bet,opp_bet),0);
      features[idx++]=std::min((float)to_call/sp,4.f); features[idx++]=std::min((float)opp_bet/sp,4.f);
      features[idx++]=std::min((float)my_bet/sp,4.f);  features[idx++]=std::min((float)mr2/sp,4.f);
      features[idx++]=mr2/100.f; }

    // [111-118] Bet counts
    if(street_bet_counts){ for(int s=0;s<4;s++){ features[idx++]=std::min(street_bet_counts[s*2+hp]/4.f,1.f); features[idx++]=std::min(street_bet_counts[s*2+1-hp]/4.f,1.f); } }
    else { for(int i=0;i<8;i++) features[idx++]=0.f; }
    // idx==119
}
