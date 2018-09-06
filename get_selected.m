function out = get_selected(x)
    [val, idx] = max(x{end});
    out = int8(idx - 1);
end