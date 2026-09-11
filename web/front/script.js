// TODO: broadcast masking into here since its reimplemented

(function(){
  "use strict";

  // ---- backend action-index mapping (must mirror BSEnv.declareActions) ----
  // declareActions = [(qty, honest) for qty in 1..4 for honest in 0..qty]
  function declareIndex(qty, honest){
    let idx = 0;
    for (let q = 1; q < qty; q++) idx += (q + 1);
    return idx + honest;
  }
  const ACTION_CHALLENGE = 14;
  const ACTION_PASS = 15;

  const RANK_DISPLAY = {
    ACE: "A", TWO: "2", THREE: "3", FOUR: "4", FIVE: "5", SIX: "6",
    SEVEN: "7", EIGHT: "8", NINE: "9", TEN: "10", JACK: "J", QUEEN: "Q", KING: "K"
  };
  const SUIT_GLYPH = { HEARTS: "♥", DIAMONDS: "♦", SPADES: "♠", CLUBS: "♣" };
  const RED_SUITS = new Set(["HEARTS", "DIAMONDS"]);

  // ---- state ----
  let ws = null;
  let myId = null;
  let obs = null;
  let awaitingUpdate = false;
  let selectedQty = null;
  let selectedHonest = null;
  let lastClaimKey = null;
  let firstHandRender = true;

  const el = (id) => document.getElementById(id);

  function log(msg, isErr){
    const box = el("log");
    const line = document.createElement("div");
    if (isErr) line.className = "err";
    const t = new Date().toLocaleTimeString();
    line.textContent = `[${t}] ${msg}`;
    box.appendChild(line);
    box.scrollTop = box.scrollHeight;
  }

  function setConnected(isConnected){
    el("statusDot").classList.toggle("live", isConnected);
    el("statusLine").lastChild
      ? (el("statusLine").childNodes[1].textContent = isConnected ? "connected" : "disconnected")
      : null;
    el("app").style.display = isConnected ? "block" : "none";
    el("connectBtn").style.display = isConnected ? "none" : "inline-block";
    el("disconnectBtn").style.display = isConnected ? "inline-block" : "none";
    el("serverUrl").disabled = isConnected;
    el("gameId").disabled = isConnected;
  }

  // ---- connection ----

  el("connectBtn").addEventListener("click", () => {
    const base = el("serverUrl").value.replace(/\/+$/, "");
    const gid = encodeURIComponent(el("gameId").value.trim() || "demo");
    const url = `${base}/ws/${gid}`;

    try{
      ws = new WebSocket(url);
    }catch(e){
      log(`Could not open ${url}: ${e.message}`, true);
      return;
    }

    ws.onopen = () => {
      setConnected(true);
      log(`Connected to ${url}`);
    };

    ws.onclose = () => {
      setConnected(false);
      myId = null;
      obs = null;
      log("Connection closed. Reconnect to keep playing.", true);
    };

    ws.onerror = () => {
      log("Websocket error — check the server address and that the server is running.", true);
    };

    ws.onmessage = (event) => {
      let data;
      try{
        data = JSON.parse(event.data);
      }catch(e){
        log(`Unparseable message: ${event.data}`, true);
        return;
      }
      // main.py's initial playerObs send double-encodes the payload
      // (json.dumps passed into send_json). Unwrap it if so.
      if (typeof data === "string"){
        try{ data = JSON.parse(data); }catch(e){ /* leave as string */ }
      }
      handleMessage(data);
    };
  });

  el("disconnectBtn").addEventListener("click", () => {
    if (ws) ws.close();
  });

  function send(type, data){
    if (!ws || ws.readyState !== WebSocket.OPEN){
      log("Not connected.", true);
      return;
    }
    ws.send(JSON.stringify({ type, data }));
  }

  // ---- message handling ----

  function handleMessage(msg){
    if (!msg || !msg.type){
      log(`Unrecognized message: ${JSON.stringify(msg)}`, true);
      return;
    }
    switch(msg.type){
      case "player_assigned":
        myId = msg.data;
        el("playerBadge").style.display = "inline-block";
        el("playerBadge").className = "player-badge mine";
        el("playerBadge").textContent = `You are Player ${myId}`;
        log(`Assigned seat ${myId}`);
        break;

      case "playerObs":
        awaitingUpdate = false;
        obs = msg.data;
        selectedQty = null;
        selectedHonest = null;
        render();
        break;

      case "error":
        awaitingUpdate = false;
        log(`Server rejected: ${msg.data}`, true);
        render();
        break;

      case "debug":
        log(`Debug: ${JSON.stringify(msg.data)}`);
        break;

      default:
        log(`Unhandled message type "${msg.type}": ${JSON.stringify(msg.data)}`);
    }
  }

  // ---- bot / debug controls ----

  el("addBotBtn").addEventListener("click", () => {
    const seat = Number(el("botSeat").value);
    const agent = el("botAgent").value.trim() || "random";
    send("add_bot", { player_id: seat, agent });
    log(`Requested bot "${agent}" for seat ${seat}`);
  });

  el("whoAmIBtn").addEventListener("click", () => {
    send("debug", "getPlayer");
  });

  // ---- rendering ----

  function render(){
    if (!obs) return;

    renderSeats();
    renderCenter();
    renderHand();
    renderDock();
  }

  function renderSeats(){
    const box = el("seats");
    box.innerHTML = "";
    const handSizes = obs.hand_sizes || [];
    handSizes.forEach((size, pid) => {
      if (pid === myId) return; // your own hand is shown below, not as a seat
      const seat = document.createElement("div");
      seat.className = "seat" + (obs.current_player === pid ? " turn" : "");

      const name = document.createElement("div");
      name.className = "name";
      name.textContent = `Player ${pid}`;
      seat.appendChild(name);

      const backs = document.createElement("div");
      backs.className = "cardbacks";
      const shown = Math.min(size, 8);
      for (let i = 0; i < shown; i++){
        const c = document.createElement("div");
        c.className = "cardback";
        backs.appendChild(c);
      }
      seat.appendChild(backs);

      const count = document.createElement("div");
      count.className = "count";
      count.textContent = `${size} card${size === 1 ? "" : "s"}`;
      seat.appendChild(count);

      box.appendChild(seat);
    });
  }

  function renderCenter(){
    const rank = obs.current_rank;
    el("rankEmblem").textContent = rank ? (RANK_DISPLAY[rank] || rank) : "—";

    const claim = obs.current_claim;
    const headline = el("claimHeadline");
    const sub = el("claimSub");

    if (claim){
      const claimerName = obs.last_actor === myId ? "You" : `Player ${obs.last_actor}`;
      headline.textContent = `${claimerName} claim${obs.last_actor === myId ? "" : "s"} ${claim.quantity} × ${RANK_DISPLAY[claim.rank] || claim.rank}`;
    } else {
      headline.textContent = "Awaiting a declare";
    }
    sub.textContent = obs.phase === "CHALLENGE" ? "Challenge phase" : "Declare phase";

    const claimKey = claim ? `${claim.quantity}-${claim.rank}-${obs.turn_number}` : `none-${obs.turn_number}`;
    if (claimKey !== lastClaimKey){
      headline.classList.remove("pulse");
      void headline.offsetWidth; // restart animation
      headline.classList.add("pulse");
      lastClaimKey = claimKey;
    }

    el("pileLabel").textContent = `Pile: ${obs.pile_size}`;
    const stack = el("pileStack");
    stack.innerHTML = "";
    const shown = Math.min(obs.pile_size, 6);
    for (let i = 0; i < shown; i++){
      const c = document.createElement("div");
      c.className = "cardback";
      c.style.top = `${i * 3}px`;
      stack.appendChild(c);
    }

    const banner = el("turnBanner");
    const myTurn = obs.current_player === myId;
    banner.textContent = myTurn ? "Your move" : `Waiting on Player ${obs.current_player}`;
    banner.className = "turn-banner" + (myTurn ? " mine" : "");
  }

  function renderHand(){
    const row = el("handRow");
    row.innerHTML = "";
    const hand = obs.player_hand || [];
    row.classList.toggle("deal-in", firstHandRender);
    hand.forEach((card, i) => {
      const tile = document.createElement("div");
      tile.className = "card" + (RED_SUITS.has(card.suit) ? " red" : "");
      tile.style.setProperty("--i", i);
      const rank = document.createElement("div");
      rank.textContent = RANK_DISPLAY[card.rank] || card.rank;
      const suit = document.createElement("div");
      suit.className = "suit";
      suit.textContent = SUIT_GLYPH[card.suit] || "";
      tile.appendChild(rank);
      tile.appendChild(suit);
      row.appendChild(tile);
    });
    firstHandRender = false;
  }

  function renderDock(){
    const dock = el("dock");
    dock.innerHTML = "";

    const myTurn = obs.current_player === myId;
    if (!myTurn){
      const note = document.createElement("div");
      note.className = "waiting-note";
      note.textContent = `Waiting on Player ${obs.current_player} to act.`;
      dock.appendChild(note);
      return;
    }

    if (obs.phase === "CHALLENGE"){
      renderChallengeDock(dock);
    } else {
      renderDeclareDock(dock);
    }
  }

  function renderChallengeDock(dock){
    const hint = document.createElement("div");
    hint.className = "hint";
    const claim = obs.current_claim;
    hint.textContent = claim
      ? `Player ${obs.last_actor} claims ${claim.quantity} × ${RANK_DISPLAY[claim.rank] || claim.rank}. Call it, or pass.`
      : "Call the last claim, or pass.";
    dock.appendChild(hint);

    const row = document.createElement("div");
    row.className = "row";

    const callBtn = document.createElement("button");
    callBtn.className = "btn danger";
    callBtn.textContent = "Call BS";
    callBtn.disabled = awaitingUpdate;
    callBtn.addEventListener("click", () => submitAction(ACTION_CHALLENGE));

    const passBtn = document.createElement("button");
    passBtn.className = "btn";
    passBtn.textContent = "Pass";
    passBtn.disabled = awaitingUpdate;
    passBtn.addEventListener("click", () => submitAction(ACTION_PASS));

    row.appendChild(callBtn);
    row.appendChild(passBtn);
    dock.appendChild(row);
  }

  function renderDeclareDock(dock){
    const handSize = (obs.player_hand || []).length;
    const matches = (obs.player_hand || []).filter(c => c.rank === obs.current_rank).length;

    const hint = document.createElement("div");
    hint.className = "hint";
    hint.textContent = `You hold ${matches} card${matches === 1 ? "" : "s"} of rank ${RANK_DISPLAY[obs.current_rank] || obs.current_rank}.`;
    dock.appendChild(hint);

    // quantity row
    const qtyRow = document.createElement("div");
    qtyRow.className = "row";
    const qtyLabel = document.createElement("span");
    qtyLabel.className = "label";
    qtyLabel.textContent = "How many";
    qtyRow.appendChild(qtyLabel);

    for (let qty = 1; qty <= 4; qty++){
      const btn = document.createElement("button");
      btn.className = "pick" + (selectedQty === qty ? " selected" : "");
      btn.textContent = qty;
      btn.disabled = qty > handSize;
      btn.addEventListener("click", () => {
        selectedQty = qty;
        if (selectedHonest !== null && selectedHonest > qty) selectedHonest = null;
        renderDock();
      });
      qtyRow.appendChild(btn);
    }
    dock.appendChild(qtyRow);

    // honest row (only once a quantity is chosen)
    if (selectedQty !== null){
      const honRow = document.createElement("div");
      honRow.className = "row";
      const honLabel = document.createElement("span");
      honLabel.className = "label";
      honLabel.textContent = "True cards";
      honRow.appendChild(honLabel);

      for (let honest = 0; honest <= selectedQty; honest++){
        const btn = document.createElement("button");
        btn.className = "pick" + (selectedHonest === honest ? " selected" : "");
        btn.textContent = honest;
        btn.disabled = honest > matches;
        btn.addEventListener("click", () => {
          selectedHonest = honest;
          renderDock();
        });
        honRow.appendChild(btn);
      }
      dock.appendChild(honRow);
    }

    if (selectedQty !== null && selectedHonest !== null){
      const preview = document.createElement("div");
      preview.className = "declare-preview";
      const bluffCount = selectedQty - selectedHonest;
      preview.textContent = bluffCount === 0
        ? `Declaring ${selectedQty} × ${RANK_DISPLAY[obs.current_rank]}, fully honest.`
        : `Declaring ${selectedQty} × ${RANK_DISPLAY[obs.current_rank]} — ${selectedHonest} true, ${bluffCount} bluffed.`;
      dock.appendChild(preview);

      const row = document.createElement("div");
      row.className = "row";
      const declareBtn = document.createElement("button");
      declareBtn.className = "btn primary";
      declareBtn.textContent = "Declare";
      declareBtn.disabled = awaitingUpdate;
      declareBtn.addEventListener("click", () => {
        submitAction(declareIndex(selectedQty, selectedHonest));
      });
      row.appendChild(declareBtn);
      dock.appendChild(row);
    }
  }

  function submitAction(action){
    awaitingUpdate = true;
    send("action", action);
    render();
  }

})();