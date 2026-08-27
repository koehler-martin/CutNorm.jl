"""
    _jump_status(status) -> Symbol

Map a JuMP/MOI `TerminationStatusCode` to the `Symbol` convention used by the
CutNorm solution types (`:optimal`, `:max_time`, otherwise the status name).
"""
function _jump_status(status)
    str = string(status)
    if str == "OPTIMAL"
        return :optimal
    elseif str == "TIME_LIMIT"
        return :max_time
    else
        return Symbol(str)
    end
end
