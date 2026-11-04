--[[
  Isaac AI Bridge
  ===============
  The game's half of the AI. Every game update (30 per second) this mod:

    1. builds a snapshot of the game (player, enemies, projectiles, room grid),
    2. sends it as one line of JSON to a Python process over TCP,
    3. WAITS for Python to reply with an action, then
    4. feeds that action to the game through MC_INPUT_ACTION, as if it came
       from a controller.

  Step 3 is "lockstep": the game cannot run ahead of the AI. If Python is busy
  (for example while PPO updates its neural network), the game simply freezes
  until the next action arrives. That keeps the training data clean.

  If no Python process is listening, the mod does nothing and you can play
  normally. It retries the connection every few seconds.

  LuaSocket is only available when the game runs with the --luadebug launch
  option. That option gives ALL enabled mods file and network access, so only
  enable mods you trust while using it.
]]

local mod = RegisterMod("Isaac AI Bridge", 1)
local json = require("json")
local game = Game()

local HOST = "127.0.0.1"
local PORT = 28771
local RECV_TIMEOUT = 60   -- seconds to wait for Python before giving up
local RETRY_FRAMES = 150  -- render frames between connection attempts (~2.5 s)

local socket = nil
do
  local ok, lib = pcall(require, "socket")
  if ok and lib then
    socket = lib
  else
    print("[AI Bridge] LuaSocket unavailable - launch the game with --luadebug")
  end
end

local conn = nil
local nextRetry = 0

-- The action Python most recently asked for.
--   mx, my : -1, 0 or 1 (screen coordinates, so my = -1 is UP)
--   shoot  : 0 none, 1 left, 2 right, 3 up, 4 down
local control = false
local action = { mx = 0, my = 0, shoot = 0, bomb = false, item = false, pill = false, drop = false }
local prevActive = {}   -- per button: was it held on the previous update (for "triggered")
local triggered = {}    -- per button: did it go from released to pressed this update

-- Arena mode = a training episode set up by reset_arena. Only then do we
-- prevent lethal damage (so training never hits a game-over screen).
local arena = false

-- While in arena mode the doors stay shut, so dodging can't carry Isaac out of
-- the room. The game may try to reopen them because the room counts as
-- "cleared", so we re-close them every update instead of only once.
local function closeDoors()
  local room = game:GetRoom()
  for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
    local door = room:GetDoor(slot)
    if door and door:IsOpen() then door:Close(true) end
  end
end

local function openDoors()
  local room = game:GetRoom()
  if not room:IsClear() then return end  -- a real fight: leave the doors to the game
  for slot = 0, DoorSlot.NUM_DOOR_SLOTS - 1 do
    local door = room:GetDoor(slot)
    if door and not door:IsOpen() then door:Open() end
  end
end

-- Single place that switches arena mode, so leaving it ALWAYS reopens the
-- doors (otherwise a crash or Ctrl+C in Python would lock you in the room).
local function setArena(on)
  if arena and not on then openDoors() end
  arena = on
  if on then closeDoors() end
end

-- Events since the last snapshot. Reset after each send.
local ev
local function resetEvents()
  ev = { hits = 0, damage_taken = 0, damage_dealt = 0, kills = 0, lethal = false }
end
resetEvents()

---------------------------------------------------------------------------
-- Connection
---------------------------------------------------------------------------

local function disconnect(reason)
  if conn then pcall(function() conn:close() end) end
  conn = nil
  control = false
  setArena(false)
  print("[AI Bridge] disconnected: " .. tostring(reason))
end

local function tryConnect()
  if conn or not socket then return end
  local now = Isaac.GetFrameCount()
  if now < nextRetry then return end
  nextRetry = now + RETRY_FRAMES

  local c = socket.tcp()
  c:settimeout(0.02)
  local ok = c:connect(HOST, PORT)
  if ok then
    c:settimeout(RECV_TIMEOUT)
    pcall(function() c:setoption("tcp-nodelay", true) end)
    conn = c
    print("[AI Bridge] connected to Python")
  else
    c:close()
  end
end

---------------------------------------------------------------------------
-- Snapshot
---------------------------------------------------------------------------

local function r(x) return math.floor(x * 100 + 0.5) / 100 end
local function vec(v) return { r(v.X), r(v.Y) } end

-- The room's collision grid as a string, one digit per cell (row-major):
-- 0 free, 1 pit, 2 object (rock, poop...), 3 solid, 4 wall, 5 wall-except-player
local function gridString(room)
  local t = {}
  for i = 0, room:GetGridSize() - 1 do
    t[#t + 1] = tostring(room:GetGridCollision(i))
  end
  return table.concat(t)
end

local function buildState()
  local room = game:GetRoom()
  local level = game:GetLevel()
  local p = Isaac.GetPlayer(0)

  local enemies, projectiles = {}, {}
  for _, e in ipairs(Isaac.GetRoomEntities()) do
    if e:IsActiveEnemy(false) and not e:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
      enemies[#enemies + 1] = {
        id = e.Index, type = e.Type, var = e.Variant, sub = e.SubType,
        pos = vec(e.Position), vel = vec(e.Velocity),
        hp = r(e.HitPoints), max_hp = r(e.MaxHitPoints), size = r(e.Size),
        boss = e:IsBoss(), vuln = e:IsVulnerableEnemy(),
      }
    elseif e.Type == EntityType.ENTITY_PROJECTILE then
      projectiles[#projectiles + 1] = { pos = vec(e.Position), vel = vec(e.Velocity), size = r(e.Size) }
    end
  end

  return {
    t = "state",
    frame = game:GetFrameCount(),
    stage = level:GetStage(),
    stage_type = level:GetStageType(),
    control = control,
    arena = arena,
    events = ev,
    room = {
      index = level:GetCurrentRoomIndex(),
      type = room:GetType(),
      shape = room:GetRoomShape(),
      clear = room:IsClear(),
      tl = vec(room:GetTopLeftPos()),
      br = vec(room:GetBottomRightPos()),
      w = room:GetGridWidth(),
      h = room:GetGridHeight(),
      grid = gridString(room),
    },
    player = {
      pos = vec(p.Position), vel = vec(p.Velocity),
      grid = room:GetGridIndex(p.Position),
      player_type = p:GetPlayerType(),
      hearts = p:GetHearts(), max_hearts = p:GetMaxHearts(), soul = p:GetSoulHearts(),
      damage = r(p.Damage), fire_delay = r(p.MaxFireDelay), speed = r(p.MoveSpeed),
      range = r(p.TearRange), shot_speed = r(p.ShotSpeed),
      coins = p:GetNumCoins(), bombs = p:GetNumBombs(), keys = p:GetNumKeys(),
      dead = p:IsDead(),
    },
    enemies = enemies,
    projectiles = projectiles,
  }
end

---------------------------------------------------------------------------
-- Training arena: wipe the room, heal, spawn a fresh set of enemies
---------------------------------------------------------------------------

local function resetArena(msg)
  local room = game:GetRoom()
  local p = Isaac.GetPlayer(0)

  for _, e in ipairs(Isaac.GetRoomEntities()) do
    if e:IsEnemy() or e.Type == EntityType.ENTITY_PROJECTILE or e.Type == EntityType.ENTITY_TEAR then
      e:Remove()
    end
  end

  p:AddHearts(24)  -- refill red hearts (capped at max)
  p.Position = room:GetCenterPos()
  p.Velocity = Vector(0, 0)

  local rng = RNG()
  local seed = tonumber(msg.seed) or Random()
  if seed < 1 then seed = 1 end
  rng:SetSeed(seed, 35)

  local minDist = tonumber(msg.min_dist) or 160
  local tl, br = room:GetTopLeftPos(), room:GetBottomRightPos()
  local function randomFreePos()
    for _ = 1, 60 do
      local pos = Vector(tl.X + rng:RandomFloat() * (br.X - tl.X), tl.Y + rng:RandomFloat() * (br.Y - tl.Y))
      if room:GetGridCollisionAtPos(pos) == GridCollisionClass.COLLISION_NONE
          and pos:Distance(p.Position) > minDist then
        return pos
      end
    end
    return Isaac.GetRandomPosition()
  end

  for _, spec in ipairs(msg.enemies or {}) do
    Isaac.Spawn(spec[1], spec[2] or 0, spec[3] or 0, randomFreePos(), Vector(0, 0), nil)
  end

  setArena(true)
end

---------------------------------------------------------------------------
-- Messages from Python
---------------------------------------------------------------------------

-- Which buttons the current action holds down.
local function buttonActive(button)
  local B = ButtonAction
  if button == B.ACTION_LEFT then return action.mx < 0 end
  if button == B.ACTION_RIGHT then return action.mx > 0 end
  if button == B.ACTION_UP then return action.my < 0 end
  if button == B.ACTION_DOWN then return action.my > 0 end
  if button == B.ACTION_SHOOTLEFT then return action.shoot == 1 end
  if button == B.ACTION_SHOOTRIGHT then return action.shoot == 2 end
  if button == B.ACTION_SHOOTUP then return action.shoot == 3 end
  if button == B.ACTION_SHOOTDOWN then return action.shoot == 4 end
  if button == B.ACTION_BOMB then return action.bomb end
  if button == B.ACTION_ITEM then return action.item end
  if button == B.ACTION_PILLCARD then return action.pill end
  if button == B.ACTION_DROP then return action.drop end
  return nil  -- not a button we control (menus, pause, map...)
end

local CONTROLLED = {
  ButtonAction.ACTION_LEFT, ButtonAction.ACTION_RIGHT, ButtonAction.ACTION_UP, ButtonAction.ACTION_DOWN,
  ButtonAction.ACTION_SHOOTLEFT, ButtonAction.ACTION_SHOOTRIGHT, ButtonAction.ACTION_SHOOTUP,
  ButtonAction.ACTION_SHOOTDOWN, ButtonAction.ACTION_BOMB, ButtonAction.ACTION_ITEM,
  ButtonAction.ACTION_PILLCARD, ButtonAction.ACTION_DROP,
}

local function applyMessage(msg)
  if msg.control ~= nil then control = msg.control == true end
  if type(msg.move) == "table" then
    action.mx = tonumber(msg.move[1]) or 0
    action.my = tonumber(msg.move[2]) or 0
  end
  if msg.shoot ~= nil then action.shoot = tonumber(msg.shoot) or 0 end
  action.bomb = msg.bomb == true
  action.item = msg.item == true
  action.pill = msg.pill == true
  action.drop = msg.drop == true

  -- "Triggered" = pressed this update but not the previous one (e.g. drop ONE bomb).
  for _, b in ipairs(CONTROLLED) do
    local now = buttonActive(b) == true
    triggered[b] = now and not prevActive[b]
    prevActive[b] = now
  end

  if msg.cmd == "reset_arena" then
    resetArena(msg)
  elseif msg.cmd == "release" then
    control = false
    setArena(false)
  end
end

---------------------------------------------------------------------------
-- Callbacks
---------------------------------------------------------------------------

-- The main loop: one snapshot out, one action in, per game update.
mod:AddCallback(ModCallbacks.MC_POST_UPDATE, function()
  tryConnect()
  if not conn then return end
  if arena then closeDoors() end

  local ok, payload = pcall(function() return json.encode(buildState()) end)
  if not ok then
    print("[AI Bridge] snapshot error: " .. tostring(payload))
    return
  end
  resetEvents()

  local sent, err = conn:send(payload .. "\n")
  if not sent then disconnect(err) return end

  local line, rerr = conn:receive("*l")
  if not line then disconnect(rerr) return end

  local ok2, msg = pcall(json.decode, line)
  if ok2 and type(msg) == "table" then
    local ok3, aerr = pcall(applyMessage, msg)
    if not ok3 then print("[AI Bridge] bad message: " .. tostring(aerr)) end
  end
end)

-- Replace the player's inputs with the AI's while it has control.
mod:AddCallback(ModCallbacks.MC_INPUT_ACTION, function(_, entity, hook, button)
  if not control or not entity or not entity:ToPlayer() then return nil end
  local active = buttonActive(button)
  if active == nil then return nil end

  if hook == InputHook.GET_ACTION_VALUE then
    return active and 1.0 or 0.0
  elseif hook == InputHook.IS_ACTION_TRIGGERED then
    return triggered[button] == true
  else -- IS_ACTION_PRESSED
    return active
  end
end)

-- Count hits on the player and damage dealt to enemies.
mod:AddCallback(ModCallbacks.MC_ENTITY_TAKE_DMG, function(_, entity, amount)
  local p = entity:ToPlayer()
  if p then
    ev.hits = ev.hits + 1
    ev.damage_taken = ev.damage_taken + amount
    -- In the arena, cancel the killing blow and report it instead, so the
    -- episode ends without a game over.
    if arena and p:GetBoneHearts() == 0 and (p:GetHearts() + p:GetSoulHearts()) <= amount then
      ev.lethal = true
      return false
    end
  elseif entity:IsActiveEnemy(false) and not entity:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
    ev.damage_dealt = ev.damage_dealt + math.min(amount, entity.HitPoints)
  end
end)

mod:AddCallback(ModCallbacks.MC_POST_NPC_DEATH, function(_, npc)
  if not npc:HasEntityFlags(EntityFlag.FLAG_FRIENDLY) then
    ev.kills = ev.kills + 1
  end
end)

-- A new run always starts outside arena mode.
mod:AddCallback(ModCallbacks.MC_POST_GAME_STARTED, function()
  arena = false  -- fresh run: nothing to reopen
end)

-- Backup: if Isaac still gets out of the room somehow, drop arena mode so the
-- new room behaves normally (its doors are never touched).
mod:AddCallback(ModCallbacks.MC_POST_NEW_ROOM, function()
  arena = false
end)
