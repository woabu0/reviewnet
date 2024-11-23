function Do(props) {
  return (
    <div className="w-full md:w-[400px] text-center">
      <img
        src={props.icon}
        alt="icon"
        className="w-16 m-auto h-16 mt-[30px] xl:mt-0 hover:scale-110 transition-all rounded-[12px]"
      />
      <h2 className="text-2xl text-black mt-8 s">{props.title}</h2>
      <p className="text-black text-base mt-3 m">{props.detail}</p>
    </div>
  );
}
export default Do;
