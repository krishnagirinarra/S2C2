while true;
do
  r=$(( ( RANDOM % 10 )  + 1 ));
  sleep $r;
  python interfere.py;
done
